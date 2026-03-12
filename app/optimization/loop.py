"""EvolutionLoop — the main optimization loop orchestrator."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from app.gateway.base import GatewayClient
from app.models.candidate import PromptCandidate
from app.models.eval_case import EvalCase
from app.models.round_summary import OptimizationRound
from app.models.score import CandidateScore
from app.mutations.engine import MutationEngine
from app.optimization.beam_search import BeamSearchSelector
from app.optimization.hill_climbing import HillClimbingSelector
from app.runner.runner import EvalRunner
from app.scoring.aggregator import aggregate_candidate_scores
from app.scoring.deterministic import ALL_DETERMINISTIC_SCORERS
from app.storage import db as storage
from app.storage.artifacts import write_jsonl

logger = logging.getLogger(__name__)


def _score_results(
    results,
    candidates: list[PromptCandidate],
    eval_cases: list[EvalCase],
    run_id: str,
    scoring_config: dict[str, Any],
    gateway: Optional[GatewayClient] = None,
) -> list[CandidateScore]:
    """Score all run results and return CandidateScore list."""
    from app.models.score import ScoreBreakdown

    case_map = {c.id: c for c in eval_cases}
    scorers_config: list[dict[str, Any]] = scoring_config.get("scorers", [])

    # Group results by candidate
    results_by_candidate: dict[str, list] = {}
    for r in results:
        results_by_candidate.setdefault(r.candidate_id, []).append(r)

    candidate_scores: list[CandidateScore] = []

    for candidate in candidates:
        cand_results = results_by_candidate.get(candidate.id, [])
        all_breakdowns: list[ScoreBreakdown] = []

        for result in cand_results:
            case = case_map.get(result.eval_case_id)
            expected = case.expected_output if case else None

            for scorer_cfg in scorers_config:
                scorer_name = scorer_cfg.get("name")
                if not scorer_name:
                    continue

                if scorer_name == "model_judge":
                    if gateway is None:
                        logger.warning("Skipping model_judge: no gateway provided")
                        continue
                    from app.scoring.judge import ModelJudgeScorer
                    scorer = ModelJudgeScorer(gateway)
                else:
                    scorer = ALL_DETERMINISTIC_SCORERS.get(scorer_name)
                    if scorer is None:
                        logger.warning("Unknown scorer '%s', skipping", scorer_name)
                        continue

                try:
                    bd = scorer.score(result, expected, scorer_cfg)
                    all_breakdowns.append(bd)
                except Exception as exc:
                    logger.error("Scorer '%s' raised error: %s", scorer_name, exc)

        cs = aggregate_candidate_scores(
            candidate_id=candidate.id,
            run_id=run_id,
            results=cand_results,
            breakdowns=all_breakdowns,
            scoring_config=scoring_config,
        )
        candidate_scores.append(cs)

    return candidate_scores


class EvolutionLoop:
    """Runs the full prompt evolution optimization loop."""

    def __init__(
        self,
        gateway: GatewayClient,
        eval_cases: list[EvalCase],
        scoring_config: dict[str, Any],
        mutation_operators: list[str],
        strategy: str = "hill_climbing",
        rounds: int = 3,
        variants_per_candidate: int = 2,
        beam_width: int = 3,
        max_tokens: int = 1024,
        output_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self._gateway = gateway
        self._eval_cases = eval_cases
        self._scoring_config = scoring_config
        self._mutation_operators = mutation_operators
        self._strategy = strategy
        self._rounds = rounds
        self._variants_per_candidate = variants_per_candidate
        self._beam_width = beam_width
        self._max_tokens = max_tokens
        self._output_dir = output_dir or Path("data/runs")
        self._db_path = db_path
        self._dry_run = dry_run

        self._mutation_engine = MutationEngine(gateway)
        self._runner = EvalRunner(
            gateway=gateway,
            max_tokens=max_tokens,
            run_dir=self._output_dir,
            dry_run=dry_run,
        )

        if strategy == "beam_search":
            self._selector = BeamSearchSelector(beam_width=beam_width)
        else:
            self._selector = HillClimbingSelector()  # type: ignore[assignment]

        self._conn = storage.get_connection(db_path) if db_path else None

    def run(self, baseline: PromptCandidate) -> tuple[PromptCandidate, list[OptimizationRound]]:
        """Run the full evolution loop starting from baseline.

        Returns:
            Tuple of (best_candidate, list[OptimizationRound]).
        """
        logger.info(
            "Starting evolution loop: strategy=%s, rounds=%d, variants/candidate=%d",
            self._strategy,
            self._rounds,
            self._variants_per_candidate,
        )

        if self._conn:
            storage.upsert_candidate(self._conn, baseline)

        current_generation: list[PromptCandidate] = [baseline]
        all_rounds: list[OptimizationRound] = []
        best_candidate = baseline

        for round_num in range(1, self._rounds + 1):
            logger.info("=== Round %d / %d ===", round_num, self._rounds)

            # Generate mutations from current generation
            new_variants: list[PromptCandidate] = []
            for parent in current_generation:
                try:
                    children = self._mutation_engine.mutate(
                        parent,
                        self._mutation_operators,
                        n=self._variants_per_candidate,
                    )
                    new_variants.extend(children)
                    if self._conn:
                        for child in children:
                            storage.upsert_candidate(self._conn, child)
                except Exception as exc:
                    logger.error("Mutation failed for candidate %s: %s", parent.id[:8], exc)

            # Evaluate all: current generation + new variants
            candidates_this_round = current_generation + new_variants
            run_id = str(uuid.uuid4())

            run_id, results = self._runner.run(
                candidates=candidates_this_round,
                eval_cases=self._eval_cases,
                run_id=run_id,
            )

            if self._conn:
                for result in results:
                    storage.insert_run_result(self._conn, result)

            # Score
            candidate_scores = _score_results(
                results=results,
                candidates=candidates_this_round,
                eval_cases=self._eval_cases,
                run_id=run_id,
                scoring_config=self._scoring_config,
                gateway=self._gateway,
            )

            if self._conn:
                for cs in candidate_scores:
                    storage.upsert_candidate_score(self._conn, cs)

            # Select top candidates for next round
            selected = self._selector.select(candidate_scores)
            if not selected:
                logger.warning("No candidates selected in round %d — keeping previous best", round_num)
                selected = [candidate_scores[0]] if candidate_scores else []

            selected_ids = [s.candidate_id for s in selected]
            best_score = selected[0] if selected else None

            # Find best candidate object
            cand_map = {c.id: c for c in candidates_this_round}
            if best_score and best_score.candidate_id in cand_map:
                best_candidate = cand_map[best_score.candidate_id]

            # Build round summary
            scores_dict = {s.candidate_id: s.aggregate_score for s in candidate_scores}
            rnd = OptimizationRound(
                round_number=round_num,
                strategy=self._strategy,
                run_id=run_id,
                candidates_evaluated=[c.id for c in candidates_this_round],
                top_candidate_ids=selected_ids,
                scores=scores_dict,
                metadata={
                    "variants_generated": len(new_variants),
                    "beam_width": self._beam_width,
                },
            )
            all_rounds.append(rnd)

            if self._conn:
                storage.insert_optimization_round(self._conn, rnd)

            # Write round summary artifact
            round_dir = self._output_dir / run_id
            round_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(round_dir / "round_summary.jsonl", [rnd.model_dump()])

            # Prepare next generation
            current_generation = [
                cand_map[sid] for sid in selected_ids if sid in cand_map
            ]
            if not current_generation:
                current_generation = [baseline]

            logger.info(
                "Round %d done. Best score: %.4f (candidate %s)",
                round_num,
                best_score.aggregate_score if best_score else 0.0,
                best_score.candidate_id[:8] if best_score else "?",
            )

        logger.info(
            "Evolution loop complete. Best candidate: %s",
            best_candidate.id[:8],
        )
        return best_candidate, all_rounds

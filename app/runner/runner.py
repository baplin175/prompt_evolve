"""EvalRunner — evaluates prompt candidates against eval cases."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from app.gateway.base import GatewayClient
from app.models.candidate import PromptCandidate
from app.models.eval_case import EvalCase
from app.models.run_result import RunResult
from app.storage.artifacts import write_run_artifacts

logger = logging.getLogger(__name__)


class EvalRunner:
    """Runs a set of prompt candidates against a set of eval cases."""

    def __init__(
        self,
        gateway: GatewayClient,
        max_tokens: int = 1024,
        run_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self._gateway = gateway
        self._max_tokens = max_tokens
        self._run_dir = run_dir
        self._dry_run = dry_run

    def run(
        self,
        candidates: list[PromptCandidate],
        eval_cases: list[EvalCase],
        run_id: Optional[str] = None,
    ) -> tuple[str, list[RunResult]]:
        """Evaluate all candidates × eval cases.

        Args:
            candidates: Prompt candidates to evaluate.
            eval_cases: Eval cases to run against.
            run_id: Optional run identifier; auto-generated if not provided.

        Returns:
            Tuple of (run_id, list[RunResult]).
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        logger.info(
            "Starting run %s: %d candidates × %d eval cases (dry_run=%s)",
            run_id[:8],
            len(candidates),
            len(eval_cases),
            self._dry_run,
        )

        results: list[RunResult] = []

        for candidate in candidates:
            for case in eval_cases:
                result = self._evaluate_one(run_id, candidate, case)
                results.append(result)

        logger.info("Run %s complete: %d results", run_id[:8], len(results))

        if self._run_dir is not None:
            run_subdir = self._run_dir / run_id
            artifacts = [r.model_dump() for r in results]
            path = write_run_artifacts(run_subdir, artifacts)
            logger.info("Artifacts written to %s", path)

        return run_id, results

    def _evaluate_one(
        self,
        run_id: str,
        candidate: PromptCandidate,
        case: EvalCase,
    ) -> RunResult:
        """Evaluate a single candidate/case pair."""
        if case.is_multi_turn:
            # Build full message list from conversation turns
            messages: list[dict[str, str]] = []
            if candidate.system_prompt:
                messages.append({"role": "system", "content": candidate.system_prompt})
            for turn in case.turns:
                content = turn.content
                if turn.role == "user":
                    try:
                        content = candidate.prompt_text.format(input=content)
                    except KeyError:
                        content = f"{candidate.prompt_text}\n\n{content}"
                messages.append({"role": turn.role, "content": content})
            # For display / fallback, derive user_content from the last user turn
            user_content = messages[-1]["content"] if messages else case.input
        else:
            messages = None  # type: ignore[assignment]
            # Build user content by substituting {input} placeholder
            try:
                user_content = candidate.prompt_text.format(input=case.input)
            except KeyError as exc:
                logger.warning(
                    "Prompt template substitution failed for candidate %s: %s",
                    candidate.id[:8],
                    exc,
                )
                user_content = f"{candidate.prompt_text}\n\n{case.input}"

        if self._dry_run:
            logger.debug("DRY RUN: skipping LLM call for case %s", case.id[:8])
            return RunResult(
                run_id=run_id,
                candidate_id=candidate.id,
                eval_case_id=case.id,
                raw_output="[DRY RUN — no LLM call made]",
                latency_ms=0.0,
            )

        try:
            response = self._gateway.complete(
                model=candidate.model,
                temperature=candidate.temperature,
                max_tokens=self._max_tokens,
                system_prompt=candidate.system_prompt,
                user_content=user_content,
                messages=messages,
            )

            # Try to parse output as JSON
            parsed = None
            try:
                parsed = json.loads(response.content.strip())
            except (json.JSONDecodeError, ValueError):
                pass

            return RunResult(
                run_id=run_id,
                candidate_id=candidate.id,
                eval_case_id=case.id,
                raw_output=response.content,
                parsed_output=parsed,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

        except Exception as exc:
            logger.error(
                "Error evaluating candidate %s on case %s: %s",
                candidate.id[:8],
                case.id[:8],
                exc,
            )
            return RunResult(
                run_id=run_id,
                candidate_id=candidate.id,
                eval_case_id=case.id,
                raw_output="",
                error=str(exc),
                latency_ms=0.0,
            )

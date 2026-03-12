"""Markdown report generator."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.models.candidate import PromptCandidate
from app.models.eval_case import EvalCase
from app.models.round_summary import OptimizationRound
from app.models.run_result import RunResult
from app.models.score import CandidateScore
from app.reporting.leaderboard import build_leaderboard


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _lineage_tree(
    candidates: list[PromptCandidate],
    root_id: str,
    indent: int = 0,
) -> list[str]:
    """Build an ASCII lineage tree rooted at root_id."""
    lines = []
    for c in candidates:
        if c.id == root_id:
            lines.append(" " * indent + f"- [{c.mutation_strategy}] {c.id[:8]}")
            # Find children
            for child in candidates:
                if child.parent_id == c.id:
                    lines.extend(_lineage_tree(candidates, child.id, indent + 2))
    return lines


def generate_report(
    run_id: str,
    scores: list[CandidateScore],
    candidates: list[PromptCandidate],
    eval_cases: list[EvalCase],
    results: list[RunResult],
    rounds: list[OptimizationRound],
    strategy: str,
    output_path: Path,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Generate a markdown report and write it to output_path.

    Returns the report content as a string.
    """
    leaderboard = build_leaderboard(scores, candidates)
    cand_map = {c.id: c for c in candidates}
    case_map = {c.id: c for c in eval_cases}

    lines: list[str] = []

    # Header
    lines += [
        "# Prompt Evolution Report",
        "",
        f"**Generated:** {_now()}",
        f"**Run ID:** `{run_id}`",
        f"**Strategy:** {strategy}",
        f"**Rounds:** {len(rounds)}",
        f"**Eval Cases:** {len(eval_cases)}",
        f"**Candidates Evaluated:** {len(candidates)}",
        "",
    ]

    if extra_metadata:
        lines += ["## Run Metadata", ""]
        lines += [f"- **{k}:** {v}" for k, v in extra_metadata.items()]
        lines += [""]

    # Leaderboard
    lines += ["## Leaderboard", ""]
    lines += ["| Rank | ID | Score | Correctness | Format | Latency | Strategy |"]
    lines += ["|------|-----|-------|------------|--------|---------|----------|"]
    for entry in leaderboard:
        dims = entry["dimension_scores"]
        lines.append(
            f"| {entry['rank']} "
            f"| `{entry['candidate_id'][:8]}` "
            f"| {entry['aggregate_score']:.4f} "
            f"| {dims.get('correctness', 0):.4f} "
            f"| {dims.get('format_compliance', 0):.4f} "
            f"| {entry['avg_latency_ms']:.0f}ms "
            f"| {entry['mutation_strategy']} |"
        )
    lines += [""]

    # Best prompt
    best = leaderboard[0] if leaderboard else None
    if best:
        best_cand = cand_map.get(best["candidate_id"])
        lines += [
            "## Best Prompt",
            "",
            f"**Candidate ID:** `{best['candidate_id']}`",
            f"**Score:** {best['aggregate_score']:.4f}",
            f"**Strategy:** {best['mutation_strategy']}",
            "",
        ]
        if best_cand:
            if best_cand.system_prompt:
                lines += [
                    "### System Prompt",
                    "",
                    "```",
                    best_cand.system_prompt,
                    "```",
                    "",
                ]
            lines += [
                "### User Prompt Template",
                "",
                "```",
                best_cand.prompt_text,
                "```",
                "",
                f"**Model:** {best_cand.model} | **Temperature:** {best_cand.temperature}",
                "",
            ]

    # Score breakdown
    if best and leaderboard:
        lines += ["## Score Breakdown (Best Candidate)", ""]
        dims = leaderboard[0]["dimension_scores"]
        lines += ["| Dimension | Score |", "|-----------|-------|"]
        for dim, val in dims.items():
            lines.append(f"| {dim} | {val:.4f} |")
        lines += [""]

    # Worst failing eval cases
    error_results = [r for r in results if r.error is not None]
    if error_results:
        worst = error_results[:3]
        lines += ["## Worst Failing Eval Cases", ""]
        for i, r in enumerate(worst, start=1):
            case = case_map.get(r.eval_case_id)
            lines += [
                f"### {i}. Case `{r.eval_case_id[:8]}` (candidate `{r.candidate_id[:8]}`)",
                "",
                f"**Input:** {case.input[:200] if case else '?'}",
                "",
                f"**Error:** {r.error}",
                "",
            ]

    # Low-scoring results (by case)
    if best and results:
        best_cand_id = best["candidate_id"]
        best_results = [r for r in results if r.candidate_id == best_cand_id]
        # Find results where we know expected but output was empty/error
        failing = [r for r in best_results if r.error or not r.raw_output.strip()]
        if failing:
            lines += ["## Failure Patterns", ""]
            error_types: dict[str, int] = {}
            for r in failing:
                key = r.error.split(":")[0] if r.error else "empty_output"
                error_types[key] = error_types.get(key, 0) + 1
            for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                lines.append(f"- **{err_type}**: {count} occurrence(s)")
            lines += [""]

    # Prompt lineage
    lines += ["## Prompt Lineage", ""]
    # Find root candidates (no parent)
    roots = [c for c in candidates if c.parent_id is None]
    for root in roots:
        tree_lines = _lineage_tree(candidates, root.id)
        lines.extend(tree_lines)
    lines += [""]

    # Round summaries
    if rounds:
        lines += ["## Optimization Rounds", ""]
        for rnd in rounds:
            best_in_round = max(rnd.scores.values(), default=0.0) if rnd.scores else 0.0
            lines.append(
                f"- **Round {rnd.round_number}** ({rnd.strategy}): "
                f"{len(rnd.candidates_evaluated)} candidates, "
                f"best score={best_in_round:.4f}"
            )
        lines += [""]

    # Recommendations
    lines += [
        "## Recommendations",
        "",
        "1. Review worst-performing eval cases and consider adding targeted examples.",
        "2. If format compliance is low, apply `improve_formatting_instructions` mutation.",
        "3. If correctness is low, apply `tighten_constraints` or `add_examples` mutation.",
        "4. If latency is high, consider `reduce_verbosity` mutation or a smaller model.",
        "5. Run additional rounds with the best candidate as the new baseline.",
        "",
    ]

    content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    return content

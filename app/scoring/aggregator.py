"""Weighted score aggregation across dimensions and eval cases."""

from __future__ import annotations

import logging
from typing import Any

from app.models.run_result import RunResult
from app.models.score import CandidateScore, ScoreBreakdown

logger = logging.getLogger(__name__)

# Default scoring dimension weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.50,
    "robustness": 0.20,
    "format_compliance": 0.20,
    "latency_efficiency": 0.05,
    "cost_efficiency": 0.05,
}


def _latency_score(latency_ms: float, max_latency_ms: float = 10_000.0) -> float:
    """Convert latency to a 0–1 score (lower latency → higher score)."""
    if latency_ms <= 0:
        return 1.0
    return max(0.0, 1.0 - (latency_ms / max_latency_ms))


def _cost_score(cost_usd: float, max_cost_usd: float = 0.10) -> float:
    """Convert cost to a 0–1 score (lower cost → higher score)."""
    if cost_usd <= 0:
        return 1.0
    return max(0.0, 1.0 - (cost_usd / max_cost_usd))


def aggregate_candidate_scores(
    candidate_id: str,
    run_id: str,
    results: list[RunResult],
    breakdowns: list[ScoreBreakdown],
    scoring_config: dict[str, Any],
) -> CandidateScore:
    """Compute an aggregate CandidateScore from all run results and breakdowns.

    Args:
        candidate_id: ID of the prompt candidate being scored.
        run_id: ID of the evaluation run.
        results: All RunResult objects for this candidate.
        breakdowns: All ScoreBreakdown objects for this candidate's results.
        scoring_config: Top-level scoring configuration dict.

    Returns:
        A CandidateScore with aggregate_score and dimension_scores.
    """
    weights: dict[str, float] = scoring_config.get("weights", DEFAULT_WEIGHTS)
    max_latency_ms: float = scoring_config.get("max_latency_ms", 10_000.0)
    max_cost_usd: float = scoring_config.get("max_cost_usd", 0.10)

    # Group breakdowns by dimension
    dim_scores: dict[str, list[float]] = {dim: [] for dim in weights}

    for bd in breakdowns:
        dim = bd.dimension
        if dim in dim_scores:
            dim_scores[dim].append(bd.raw_score)

    # Compute robustness from error rate
    error_count = sum(1 for r in results if r.error is not None)
    total = len(results) if results else 1
    robustness = 1.0 - (error_count / total)
    dim_scores["robustness"].append(robustness)

    # Compute latency/cost efficiency from results
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    dim_scores["latency_efficiency"].append(_latency_score(avg_latency, max_latency_ms))

    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    total_cost = sum(costs) if costs else None
    if total_cost is not None:
        dim_scores["cost_efficiency"].append(_cost_score(total_cost, max_cost_usd))

    # Average within each dimension
    averaged_dims: dict[str, float] = {}
    for dim, scores_list in dim_scores.items():
        averaged_dims[dim] = sum(scores_list) / len(scores_list) if scores_list else 0.0

    # Weighted aggregate
    total_weight = sum(weights.values())
    aggregate = sum(
        averaged_dims.get(dim, 0.0) * w for dim, w in weights.items()
    ) / (total_weight if total_weight > 0 else 1.0)

    return CandidateScore(
        candidate_id=candidate_id,
        run_id=run_id,
        aggregate_score=round(aggregate, 6),
        dimension_scores={dim: round(v, 6) for dim, v in averaged_dims.items()},
        eval_case_count=total,
        error_count=error_count,
        avg_latency_ms=round(avg_latency, 2),
        total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
        breakdowns=breakdowns,
    )

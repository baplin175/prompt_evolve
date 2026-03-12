"""Score models — per-scorer breakdown and aggregate candidate score."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ScoreBreakdown(BaseModel):
    """Score produced by a single scorer for one run result."""

    id: str = Field(default_factory=_new_uuid)
    run_result_id: str
    scorer_name: str
    dimension: str
    raw_score: float  # 0.0 – 1.0
    weight: float = 1.0
    weighted_score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class CandidateScore(BaseModel):
    """Aggregated score for a candidate across all eval cases and dimensions."""

    candidate_id: str
    run_id: str
    aggregate_score: float  # 0.0 – 1.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    eval_case_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    total_cost_usd: Optional[float] = None
    breakdowns: list[ScoreBreakdown] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

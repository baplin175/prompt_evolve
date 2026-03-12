"""OptimizationRound — summary of one evolution loop round."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimizationRound(BaseModel):
    """Summary of one round of the evolution loop."""

    round_number: int
    strategy: str
    run_id: str
    candidates_evaluated: list[str] = Field(default_factory=list)
    top_candidate_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow)

    model_config = {"extra": "ignore"}

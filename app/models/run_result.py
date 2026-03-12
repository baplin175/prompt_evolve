"""RunResult — the raw result of evaluating one prompt candidate on one eval case."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


class RunResult(BaseModel):
    """Raw result of evaluating one prompt candidate on one eval case."""

    id: str = Field(default_factory=_new_uuid)
    run_id: str
    candidate_id: str
    eval_case_id: str
    raw_output: str = ""
    parsed_output: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    created_at: str = Field(default_factory=_utcnow)

    model_config = {"extra": "ignore"}

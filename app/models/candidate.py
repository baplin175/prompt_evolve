"""PromptCandidate — a versioned prompt with all associated parameters."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


class PromptCandidate(BaseModel):
    """A versioned prompt candidate with its generation metadata."""

    id: str = Field(default_factory=_new_uuid)
    parent_id: Optional[str] = None
    prompt_text: str
    system_prompt: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)
    mutation_strategy: str = "baseline"
    notes: str = ""
    created_at: str = Field(default_factory=_utcnow)

    model_config = {"extra": "ignore"}

"""EvalCase — a single evaluation test case."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


def _new_uuid() -> str:
    return str(uuid.uuid4())


class EvalCase(BaseModel):
    """A single evaluation test case."""

    id: str = Field(default_factory=_new_uuid)
    input: str
    expected_output: Optional[str] = None
    reference_output: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}

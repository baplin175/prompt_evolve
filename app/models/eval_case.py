"""EvalCase — a single evaluation test case."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    role: str  # "user" or "assistant"
    content: str


class EvalCase(BaseModel):
    """A single evaluation test case.

    For single-turn cases, ``input`` holds the user message.
    For multi-turn cases, ``turns`` contains the conversation history
    (alternating user/assistant messages) and ``input`` is derived from
    the last user turn or may be set to a short description.
    """

    id: str = Field(default_factory=_new_uuid)
    input: str
    expected_output: Optional[str] = None
    reference_output: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)
    turns: list[ConversationTurn] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    @property
    def is_multi_turn(self) -> bool:
        """Return True if this case has multi-turn conversation history."""
        return len(self.turns) > 0

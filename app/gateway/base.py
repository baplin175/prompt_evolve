"""Abstract gateway interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GatewayResponse:
    """Response from an LLM gateway call."""

    content: str
    model: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)


class GatewayClient(ABC):
    """Abstract base for all LLM gateway clients."""

    @abstractmethod
    def complete(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        user_content: str,
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> GatewayResponse:
        """Send a completion request and return a structured response.

        When *messages* is provided it contains the full conversation
        history (list of ``{"role": ..., "content": ...}`` dicts) and
        takes precedence over *system_prompt* / *user_content*.
        """
        ...

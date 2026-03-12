"""Abstract gateway interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


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
    ) -> GatewayResponse:
        """Send a completion request and return a structured response."""
        ...

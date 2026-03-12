"""MutationOperator protocol — the interface all operators implement."""

from __future__ import annotations

from typing import Protocol

from app.gateway.base import GatewayClient
from app.models.candidate import PromptCandidate


class MutationOperator(Protocol):
    """Protocol for prompt mutation operators."""

    name: str

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        """Produce a mutated child of *candidate*.

        Must return a new PromptCandidate with:
        - parent_id = candidate.id
        - mutation_strategy = self.name
        - all other fields set appropriately
        """
        ...

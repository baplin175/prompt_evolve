"""MutationEngine — orchestrates applying operators to produce child candidates."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from app.gateway.base import GatewayClient
from app.models.candidate import PromptCandidate
from app.mutations.operators import ALL_OPERATORS

logger = logging.getLogger(__name__)


class MutationEngine:
    """Applies one or more mutation operators to a prompt candidate."""

    def __init__(self, gateway: GatewayClient) -> None:
        self._gateway = gateway

    def mutate(
        self,
        candidate: PromptCandidate,
        operators: Sequence[str],
        n: int = 1,
    ) -> list[PromptCandidate]:
        """Generate *n* mutated children using the given operators (cycling if needed).

        Args:
            candidate: Source prompt candidate.
            operators: List of operator names to apply.
            n: Total number of child candidates to produce.

        Returns:
            List of new PromptCandidate objects (not yet persisted).
        """
        if not operators:
            raise ValueError("At least one operator must be specified")

        unknown = [op for op in operators if op not in ALL_OPERATORS]
        if unknown:
            raise ValueError(f"Unknown mutation operators: {unknown}. Available: {list(ALL_OPERATORS)}")

        children: list[PromptCandidate] = []
        for i in range(n):
            op_name = operators[i % len(operators)]
            operator = ALL_OPERATORS[op_name]
            logger.info(
                "Applying mutation '%s' to candidate %s (child %d/%d)",
                op_name,
                candidate.id[:8],
                i + 1,
                n,
            )
            try:
                child = operator.mutate(candidate, self._gateway)
                children.append(child)
            except Exception as exc:
                logger.error("Mutation '%s' failed: %s", op_name, exc)
                raise

        return children

    @staticmethod
    def available_operators() -> list[str]:
        """Return the names of all registered mutation operators."""
        return list(ALL_OPERATORS.keys())

"""Scorer protocol — the interface all scorers implement."""

from __future__ import annotations

from typing import Any, Protocol

from app.models.run_result import RunResult
from app.models.score import ScoreBreakdown


class Scorer(Protocol):
    """Protocol for all scorer implementations."""

    name: str
    dimension: str

    def score(
        self,
        result: RunResult,
        expected_output: str | None,
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        """Score a single RunResult.

        Args:
            result: The raw run result to score.
            expected_output: The expected output from the eval case (may be None).
            config: Scorer-specific configuration dict.

        Returns:
            A ScoreBreakdown with raw_score in [0.0, 1.0].
        """
        ...

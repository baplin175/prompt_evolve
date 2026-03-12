"""Hill-climbing selection: keep top-1 candidate per round."""

from __future__ import annotations

import logging

from app.models.score import CandidateScore

logger = logging.getLogger(__name__)


class HillClimbingSelector:
    """Keeps the single best-scoring candidate for the next round."""

    name = "hill_climbing"

    def select(self, scores: list[CandidateScore]) -> list[CandidateScore]:
        """Return a list containing only the best candidate.

        Args:
            scores: Scored candidates from the current round.

        Returns:
            List with at most one CandidateScore (the best).
        """
        if not scores:
            logger.warning("HillClimbingSelector: no scores provided")
            return []

        ranked = sorted(scores, key=lambda s: s.aggregate_score, reverse=True)
        best = ranked[0]
        logger.info(
            "Hill climbing selected: candidate %s (score=%.4f)",
            best.candidate_id[:8],
            best.aggregate_score,
        )
        return [best]

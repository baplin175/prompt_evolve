"""Beam-search selection: keep top-K candidates per round."""

from __future__ import annotations

import logging

from app.models.score import CandidateScore

logger = logging.getLogger(__name__)


class BeamSearchSelector:
    """Keeps the top-K scoring candidates for the next round."""

    name = "beam_search"

    def __init__(self, beam_width: int = 3) -> None:
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        self.beam_width = beam_width

    def select(self, scores: list[CandidateScore]) -> list[CandidateScore]:
        """Return the top-K candidates by aggregate score.

        Args:
            scores: Scored candidates from the current round.

        Returns:
            Up to beam_width CandidateScore objects.
        """
        if not scores:
            logger.warning("BeamSearchSelector: no scores provided")
            return []

        ranked = sorted(scores, key=lambda s: s.aggregate_score, reverse=True)
        selected = ranked[: self.beam_width]
        logger.info(
            "Beam search selected %d candidates (beam_width=%d): scores=%s",
            len(selected),
            self.beam_width,
            [f"{s.candidate_id[:8]}={s.aggregate_score:.4f}" for s in selected],
        )
        return selected

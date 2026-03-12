"""Tests for hill climbing and beam search selectors."""

from __future__ import annotations

import pytest

from app.models.score import CandidateScore
from app.optimization.beam_search import BeamSearchSelector
from app.optimization.hill_climbing import HillClimbingSelector


def _score(candidate_id: str, score: float) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        run_id="test-run",
        aggregate_score=score,
    )


class TestHillClimbingSelector:
    selector = HillClimbingSelector()

    def test_returns_single_best(self):
        scores = [_score("a", 0.5), _score("b", 0.9), _score("c", 0.3)]
        selected = self.selector.select(scores)
        assert len(selected) == 1
        assert selected[0].candidate_id == "b"

    def test_empty_scores(self):
        selected = self.selector.select([])
        assert selected == []

    def test_single_candidate(self):
        scores = [_score("only", 0.42)]
        selected = self.selector.select(scores)
        assert len(selected) == 1
        assert selected[0].candidate_id == "only"

    def test_ties_returns_one(self):
        scores = [_score("a", 0.8), _score("b", 0.8)]
        selected = self.selector.select(scores)
        assert len(selected) == 1

    def test_name(self):
        assert self.selector.name == "hill_climbing"


class TestBeamSearchSelector:
    def test_returns_top_k(self):
        scores = [_score("a", 0.5), _score("b", 0.9), _score("c", 0.3), _score("d", 0.7)]
        selector = BeamSearchSelector(beam_width=2)
        selected = selector.select(scores)
        assert len(selected) == 2
        ids = [s.candidate_id for s in selected]
        assert "b" in ids
        assert "d" in ids

    def test_beam_width_larger_than_candidates(self):
        scores = [_score("a", 0.5), _score("b", 0.9)]
        selector = BeamSearchSelector(beam_width=5)
        selected = selector.select(scores)
        assert len(selected) == 2  # Returns all available

    def test_beam_width_one_equals_hill_climbing(self):
        scores = [_score("a", 0.5), _score("b", 0.9), _score("c", 0.3)]
        selector = BeamSearchSelector(beam_width=1)
        selected = selector.select(scores)
        assert len(selected) == 1
        assert selected[0].candidate_id == "b"

    def test_invalid_beam_width(self):
        with pytest.raises(ValueError):
            BeamSearchSelector(beam_width=0)

    def test_empty_scores(self):
        selector = BeamSearchSelector(beam_width=3)
        selected = selector.select([])
        assert selected == []

    def test_name(self):
        selector = BeamSearchSelector()
        assert selector.name == "beam_search"

    def test_ordering(self):
        scores = [_score("a", 0.3), _score("b", 0.9), _score("c", 0.6)]
        selector = BeamSearchSelector(beam_width=3)
        selected = selector.select(scores)
        # Should be sorted by score descending
        assert selected[0].candidate_id == "b"
        assert selected[1].candidate_id == "c"
        assert selected[2].candidate_id == "a"

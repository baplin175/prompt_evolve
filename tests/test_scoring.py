"""Tests for all deterministic scorers."""

from __future__ import annotations

import json
import uuid

import pytest

from app.models.run_result import RunResult
from app.scoring.deterministic import (
    ExactMatchScorer,
    JsonFieldPresenceScorer,
    LengthConstraintScorer,
    RegexMatchScorer,
    SubstringMatchScorer,
)


def _result(raw_output: str) -> RunResult:
    return RunResult(
        id=str(uuid.uuid4()),
        run_id="test",
        candidate_id="cand-1",
        eval_case_id="case-1",
        raw_output=raw_output,
        latency_ms=100.0,
    )


# ---------------------------------------------------------------------------
# ExactMatchScorer
# ---------------------------------------------------------------------------


class TestExactMatchScorer:
    scorer = ExactMatchScorer()

    def test_exact_match_hit(self):
        r = _result("Paris")
        bd = self.scorer.score(r, "Paris", {})
        assert bd.raw_score == 1.0

    def test_exact_match_miss(self):
        r = _result("London")
        bd = self.scorer.score(r, "Paris", {})
        assert bd.raw_score == 0.0

    def test_exact_match_case_insensitive(self):
        r = _result("paris")
        bd = self.scorer.score(r, "Paris", {"case_sensitive": False})
        assert bd.raw_score == 1.0

    def test_exact_match_case_sensitive_miss(self):
        r = _result("paris")
        bd = self.scorer.score(r, "Paris", {"case_sensitive": True})
        assert bd.raw_score == 0.0

    def test_exact_match_strips_whitespace(self):
        r = _result("  Paris  ")
        bd = self.scorer.score(r, "Paris", {})
        assert bd.raw_score == 1.0

    def test_exact_match_no_expected(self):
        r = _result("anything")
        bd = self.scorer.score(r, None, {})
        assert bd.raw_score == 0.0

    def test_weight_applied(self):
        r = _result("Paris")
        bd = self.scorer.score(r, "Paris", {"weight": 0.5})
        assert bd.weight == 0.5
        assert bd.weighted_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SubstringMatchScorer
# ---------------------------------------------------------------------------


class TestSubstringMatchScorer:
    scorer = SubstringMatchScorer()

    def test_substring_found(self):
        r = _result("The capital is Paris, France.")
        bd = self.scorer.score(r, "Paris", {})
        assert bd.raw_score == 1.0

    def test_substring_not_found(self):
        r = _result("The capital is London.")
        bd = self.scorer.score(r, "Paris", {})
        assert bd.raw_score == 0.0

    def test_substring_case_insensitive(self):
        r = _result("the capital is paris.")
        bd = self.scorer.score(r, "Paris", {"case_sensitive": False})
        assert bd.raw_score == 1.0

    def test_no_expected(self):
        r = _result("anything")
        bd = self.scorer.score(r, None, {})
        assert bd.raw_score == 0.0


# ---------------------------------------------------------------------------
# RegexMatchScorer
# ---------------------------------------------------------------------------


class TestRegexMatchScorer:
    scorer = RegexMatchScorer()

    def test_regex_match(self):
        r = _result("The answer is 42.")
        bd = self.scorer.score(r, None, {"pattern": r"\d+"})
        assert bd.raw_score == 1.0

    def test_regex_no_match(self):
        r = _result("No numbers here.")
        bd = self.scorer.score(r, None, {"pattern": r"\d+"})
        assert bd.raw_score == 0.0

    def test_no_pattern(self):
        r = _result("something")
        bd = self.scorer.score(r, None, {})
        assert bd.raw_score == 0.0

    def test_invalid_pattern(self):
        r = _result("something")
        bd = self.scorer.score(r, None, {"pattern": "[invalid"})
        assert bd.raw_score == 0.0

    def test_case_insensitive(self):
        r = _result("hello WORLD")
        bd = self.scorer.score(r, None, {"pattern": "hello world", "ignore_case": True})
        assert bd.raw_score == 1.0


# ---------------------------------------------------------------------------
# JsonFieldPresenceScorer
# ---------------------------------------------------------------------------


class TestJsonFieldPresenceScorer:
    scorer = JsonFieldPresenceScorer()

    def test_all_fields_present(self):
        r = _result('{"name": "Alice", "age": 30}')
        bd = self.scorer.score(r, None, {"required_fields": ["name", "age"]})
        assert bd.raw_score == 1.0

    def test_some_fields_present(self):
        r = _result('{"name": "Alice"}')
        bd = self.scorer.score(r, None, {"required_fields": ["name", "age"]})
        assert bd.raw_score == pytest.approx(0.5)

    def test_no_fields_present(self):
        r = _result('{"foo": "bar"}')
        bd = self.scorer.score(r, None, {"required_fields": ["name", "age"]})
        assert bd.raw_score == 0.0

    def test_invalid_json(self):
        r = _result("not json")
        bd = self.scorer.score(r, None, {"required_fields": ["name"]})
        assert bd.raw_score == 0.0

    def test_json_array_not_object(self):
        r = _result("[1, 2, 3]")
        bd = self.scorer.score(r, None, {"required_fields": ["name"]})
        assert bd.raw_score == 0.0

    def test_markdown_fenced_json(self):
        r = _result('```json\n{"name": "Alice", "age": 30}\n```')
        bd = self.scorer.score(r, None, {"required_fields": ["name", "age"]})
        assert bd.raw_score == 1.0

    def test_no_required_fields_returns_one(self):
        r = _result('{"anything": "here"}')
        bd = self.scorer.score(r, None, {"required_fields": []})
        assert bd.raw_score == 1.0


# ---------------------------------------------------------------------------
# LengthConstraintScorer
# ---------------------------------------------------------------------------


class TestLengthConstraintScorer:
    scorer = LengthConstraintScorer()

    def test_within_bounds(self):
        r = _result("Hello world!")
        bd = self.scorer.score(r, None, {"min_chars": 5, "max_chars": 100})
        assert bd.raw_score == 1.0

    def test_too_short(self):
        r = _result("Hi")
        bd = self.scorer.score(r, None, {"min_chars": 10, "max_chars": 100})
        assert bd.raw_score == 0.0

    def test_too_long(self):
        r = _result("A" * 101)
        bd = self.scorer.score(r, None, {"min_chars": 0, "max_chars": 100})
        assert bd.raw_score == 0.0

    def test_no_max(self):
        r = _result("A" * 10000)
        bd = self.scorer.score(r, None, {"min_chars": 1})
        assert bd.raw_score == 1.0

    def test_empty_output_fails_min(self):
        r = _result("")
        bd = self.scorer.score(r, None, {"min_chars": 1})
        assert bd.raw_score == 0.0

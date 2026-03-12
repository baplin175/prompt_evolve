"""Deterministic scorers: exact match, substring, regex, JSON field, length, custom."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from typing import Any

from app.models.run_result import RunResult
from app.models.score import ScoreBreakdown

logger = logging.getLogger(__name__)


def _make_breakdown(
    result: RunResult,
    scorer_name: str,
    dimension: str,
    raw_score: float,
    weight: float = 1.0,
    details: dict | None = None,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        run_result_id=result.id,
        scorer_name=scorer_name,
        dimension=dimension,
        raw_score=raw_score,
        weight=weight,
        weighted_score=raw_score * weight,
        details=details or {},
    )


# ---------------------------------------------------------------------------
# ExactMatchScorer
# ---------------------------------------------------------------------------


class ExactMatchScorer:
    """Scores 1.0 if output exactly matches expected_output, else 0.0."""

    name = "exact_match"
    dimension = "correctness"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        weight = config.get("weight", 1.0)
        if expected_output is None:
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"reason": "no expected_output"})

        case_sensitive = config.get("case_sensitive", True)
        actual = result.raw_output
        expected = expected_output

        if not case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        actual = actual.strip()
        expected = expected.strip()

        match = actual == expected
        return _make_breakdown(result, self.name, self.dimension, 1.0 if match else 0.0, weight,
                               {"case_sensitive": case_sensitive, "match": match})


# ---------------------------------------------------------------------------
# SubstringMatchScorer
# ---------------------------------------------------------------------------


class SubstringMatchScorer:
    """Scores 1.0 if expected_output is a substring of the actual output."""

    name = "substring_match"
    dimension = "correctness"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        weight = config.get("weight", 1.0)
        if expected_output is None:
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"reason": "no expected_output"})

        case_sensitive = config.get("case_sensitive", True)
        actual = result.raw_output
        expected = expected_output

        if not case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        match = expected.strip() in actual
        return _make_breakdown(result, self.name, self.dimension, 1.0 if match else 0.0, weight,
                               {"case_sensitive": case_sensitive, "match": match})


# ---------------------------------------------------------------------------
# RegexMatchScorer
# ---------------------------------------------------------------------------


class RegexMatchScorer:
    """Scores 1.0 if the output matches a regex pattern from config."""

    name = "regex_match"
    dimension = "format_compliance"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,  # noqa: ARG002
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        weight = config.get("weight", 1.0)
        pattern = config.get("pattern")
        if not pattern:
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"reason": "no pattern in config"})

        flags = re.IGNORECASE if config.get("ignore_case", False) else 0
        try:
            match = bool(re.search(pattern, result.raw_output, flags))
        except re.error as exc:
            logger.warning("Invalid regex pattern '%s': %s", pattern, exc)
            match = False

        return _make_breakdown(result, self.name, self.dimension, 1.0 if match else 0.0, weight,
                               {"pattern": pattern, "match": match})


# ---------------------------------------------------------------------------
# JsonFieldPresenceScorer
# ---------------------------------------------------------------------------


class JsonFieldPresenceScorer:
    """Scores based on presence of required JSON fields in the output."""

    name = "json_field_presence"
    dimension = "format_compliance"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,  # noqa: ARG002
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        weight = config.get("weight", 1.0)
        required_fields: list[str] = config.get("required_fields", [])

        if not required_fields:
            return _make_breakdown(result, self.name, self.dimension, 1.0, weight,
                                   {"reason": "no required_fields specified"})

        raw = result.raw_output.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"reason": "output is not valid JSON"})

        if not isinstance(parsed, dict):
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"reason": "output is not a JSON object"})

        present = [f for f in required_fields if f in parsed]
        score = len(present) / len(required_fields)
        return _make_breakdown(result, self.name, self.dimension, score, weight,
                               {"required": required_fields, "present": present,
                                "missing": [f for f in required_fields if f not in parsed]})


# ---------------------------------------------------------------------------
# LengthConstraintScorer
# ---------------------------------------------------------------------------


class LengthConstraintScorer:
    """Scores 1.0 if output length is within [min_chars, max_chars]."""

    name = "length_constraint"
    dimension = "format_compliance"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,  # noqa: ARG002
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        weight = config.get("weight", 1.0)
        min_chars: int = config.get("min_chars", 0)
        max_chars: int | None = config.get("max_chars")

        length = len(result.raw_output)
        within_min = length >= min_chars
        within_max = (max_chars is None) or (length <= max_chars)
        ok = within_min and within_max

        return _make_breakdown(result, self.name, self.dimension, 1.0 if ok else 0.0, weight,
                               {"length": length, "min_chars": min_chars,
                                "max_chars": max_chars, "ok": ok})


# ---------------------------------------------------------------------------
# CustomPythonScorer
# ---------------------------------------------------------------------------


class CustomPythonScorer:
    """Loads and runs a user-supplied Python scorer function.

    The config must include ``module_path`` (path to .py file) and
    ``function_name`` (callable that accepts ``(result, expected_output, config)``
    and returns a float in [0.0, 1.0]).
    """

    name = "custom_python"
    dimension = "correctness"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        weight = config.get("weight", 1.0)
        module_path: str | None = config.get("module_path")
        function_name: str | None = config.get("function_name")

        if not module_path or not function_name:
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"reason": "module_path or function_name missing"})

        try:
            spec = importlib.util.spec_from_file_location("_custom_scorer", module_path)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules["_custom_scorer"] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            fn = getattr(module, function_name)
            raw_score = float(fn(result, expected_output, config))
            raw_score = max(0.0, min(1.0, raw_score))
        except Exception as exc:
            logger.error("CustomPythonScorer error: %s", exc)
            return _make_breakdown(result, self.name, self.dimension, 0.0, weight,
                                   {"error": str(exc)})

        return _make_breakdown(result, self.name, self.dimension, raw_score, weight,
                               {"module_path": module_path, "function_name": function_name})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_DETERMINISTIC_SCORERS: dict[str, object] = {
    "exact_match": ExactMatchScorer(),
    "substring_match": SubstringMatchScorer(),
    "regex_match": RegexMatchScorer(),
    "json_field_presence": JsonFieldPresenceScorer(),
    "length_constraint": LengthConstraintScorer(),
    "custom_python": CustomPythonScorer(),
}

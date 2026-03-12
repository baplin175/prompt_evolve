"""Tests for mutation operators."""

from __future__ import annotations

import random
import uuid

import pytest

from app.models.candidate import PromptCandidate
from app.mutations.engine import MutationEngine
from app.mutations.operators import VaryModelParameters, ALL_OPERATORS


def _candidate(**kwargs) -> PromptCandidate:
    defaults = dict(
        id=str(uuid.uuid4()),
        prompt_text="Answer the question: {input}",
        model="gpt-4o-mini",
        temperature=0.7,
        mutation_strategy="baseline",
    )
    defaults.update(kwargs)
    return PromptCandidate(**defaults)


class TestVaryModelParameters:
    """Test the non-LLM vary_model_parameters operator."""

    def test_produces_child(self):
        op = VaryModelParameters(rng=random.Random(42))
        candidate = _candidate()
        child = op.mutate(candidate, None)  # gateway unused
        assert child.parent_id == candidate.id
        assert child.mutation_strategy == "vary_model_parameters"

    def test_temperature_within_bounds(self):
        op = VaryModelParameters(rng=random.Random(42))
        candidate = _candidate(temperature=0.0)
        for _ in range(20):
            child = op.mutate(candidate, None)
            assert 0.0 <= child.temperature <= 2.0

    def test_high_temperature_stays_in_bounds(self):
        op = VaryModelParameters(rng=random.Random(42))
        candidate = _candidate(temperature=2.0)
        for _ in range(20):
            child = op.mutate(candidate, None)
            assert 0.0 <= child.temperature <= 2.0

    def test_prompt_text_unchanged(self):
        op = VaryModelParameters(rng=random.Random(42))
        candidate = _candidate()
        child = op.mutate(candidate, None)
        assert child.prompt_text == candidate.prompt_text

    def test_reproducible_with_seed(self):
        candidate = _candidate()
        op1 = VaryModelParameters(rng=random.Random(99))
        op2 = VaryModelParameters(rng=random.Random(99))
        child1 = op1.mutate(candidate, None)
        child2 = op2.mutate(candidate, None)
        assert child1.temperature == child2.temperature


class TestMutationEngine:
    def test_available_operators(self):
        ops = MutationEngine.available_operators()
        assert "vary_model_parameters" in ops
        assert "simplify_wording" in ops
        assert "add_examples" in ops
        assert len(ops) == 8

    def test_unknown_operator_raises(self, mock_gateway):
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        with pytest.raises(ValueError, match="Unknown mutation operators"):
            engine.mutate(candidate, ["nonexistent_operator"])

    def test_empty_operators_raises(self, mock_gateway):
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        with pytest.raises(ValueError, match="At least one operator"):
            engine.mutate(candidate, [])

    def test_vary_model_parameters_no_llm_call(self, mock_gateway):
        """VaryModelParameters should NOT call the gateway."""
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        children = engine.mutate(candidate, ["vary_model_parameters"], n=3)
        assert len(children) == 3
        mock_gateway.complete.assert_not_called()

    def test_generates_n_children(self, mock_gateway):
        """Should generate exactly n children."""
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        children = engine.mutate(candidate, ["vary_model_parameters"], n=5)
        assert len(children) == 5

    def test_cycles_operators(self, mock_gateway):
        """Should cycle through operators when n > len(operators)."""
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        children = engine.mutate(candidate, ["vary_model_parameters"], n=3)
        assert all(c.mutation_strategy == "vary_model_parameters" for c in children)

    def test_llm_mutation_uses_gateway(self, mock_gateway):
        """LLM-based mutations should call gateway.complete."""
        mock_gateway.complete.return_value.content = "Improved prompt text: {input}"
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        children = engine.mutate(candidate, ["simplify_wording"], n=1)
        assert len(children) == 1
        mock_gateway.complete.assert_called_once()

    def test_child_has_correct_parent_id(self, mock_gateway):
        mock_gateway.complete.return_value.content = "Better prompt: {input}"
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        children = engine.mutate(candidate, ["simplify_wording"], n=1)
        assert children[0].parent_id == candidate.id

    def test_mutation_error_propagates(self, mock_gateway):
        """If gateway raises, the error should propagate after retries."""
        mock_gateway.complete.side_effect = RuntimeError("API down")
        engine = MutationEngine(mock_gateway)
        candidate = _candidate()
        with pytest.raises(RuntimeError):
            engine.mutate(candidate, ["simplify_wording"], n=1)

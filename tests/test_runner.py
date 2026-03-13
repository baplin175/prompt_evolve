"""Tests for EvalRunner with mocked gateway."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.gateway.base import GatewayResponse
from app.models.candidate import PromptCandidate
from app.models.eval_case import ConversationTurn, EvalCase
from app.runner.runner import EvalRunner


def _candidate(**kwargs) -> PromptCandidate:
    defaults = dict(
        id=str(uuid.uuid4()),
        prompt_text="Answer: {input}",
        model="gpt-4o-mini",
        temperature=0.7,
        mutation_strategy="baseline",
    )
    defaults.update(kwargs)
    return PromptCandidate(**defaults)


def _case(**kwargs) -> EvalCase:
    defaults = dict(
        id=str(uuid.uuid4()),
        input="What is 2+2?",
        expected_output="4",
    )
    defaults.update(kwargs)
    return EvalCase(**defaults)


class TestEvalRunner:
    def test_run_returns_run_id_and_results(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway, max_tokens=256)
        candidates = [_candidate()]
        cases = [_case()]
        run_id, results = runner.run(candidates, cases)

        assert isinstance(run_id, str)
        assert len(results) == 1

    def test_results_have_correct_ids(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        cand = _candidate()
        case = _case()
        run_id, results = runner.run([cand], [case])

        assert results[0].candidate_id == cand.id
        assert results[0].eval_case_id == case.id
        assert results[0].run_id == run_id

    def test_multiple_candidates_and_cases(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        candidates = [_candidate() for _ in range(3)]
        cases = [_case() for _ in range(4)]
        _, results = runner.run(candidates, cases)

        assert len(results) == 12  # 3 × 4

    def test_gateway_called_correctly(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway, max_tokens=512)
        cand = _candidate(system_prompt="You are an expert.")
        case = _case(input="Hello")
        runner.run([cand], [case])

        mock_gateway.complete.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512,
            system_prompt="You are an expert.",
            user_content="Answer: Hello",
            messages=None,
        )

    def test_gateway_error_stored_in_result(self, mock_gateway):
        mock_gateway.complete.side_effect = Exception("Network error")
        runner = EvalRunner(gateway=mock_gateway)
        _, results = runner.run([_candidate()], [_case()])

        assert results[0].error == "Network error"
        assert results[0].raw_output == ""

    def test_dry_run_skips_gateway(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway, dry_run=True)
        _, results = runner.run([_candidate()], [_case()])

        mock_gateway.complete.assert_not_called()
        assert "[DRY RUN" in results[0].raw_output

    def test_artifacts_written(self, mock_gateway, tmp_path):
        runner = EvalRunner(gateway=mock_gateway, run_dir=tmp_path, dry_run=True)
        _, results = runner.run([_candidate()], [_case()])

        # Should have created a results.jsonl under tmp_path/run_id/
        jsonl_files = list(tmp_path.rglob("results.jsonl"))
        assert len(jsonl_files) == 1

    def test_custom_run_id(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        custom_id = "my-run-123"
        run_id, results = runner.run([_candidate()], [_case()], run_id=custom_id)

        assert run_id == custom_id
        assert results[0].run_id == custom_id

    def test_prompt_template_substitution(self, mock_gateway):
        """Verify that {input} is substituted with the eval case input."""
        runner = EvalRunner(gateway=mock_gateway)
        cand = _candidate(prompt_text="Tell me about {input}")
        case = _case(input="the moon")
        runner.run([cand], [case])

        call_kwargs = mock_gateway.complete.call_args.kwargs
        assert "the moon" in call_kwargs["user_content"]

    def test_latency_captured(self, mock_gateway):
        mock_gateway.complete.return_value = GatewayResponse(
            content="test",
            model="gpt-4o-mini",
            latency_ms=350.5,
            input_tokens=10,
            output_tokens=5,
        )
        runner = EvalRunner(gateway=mock_gateway)
        _, results = runner.run([_candidate()], [_case()])

        assert results[0].latency_ms == pytest.approx(350.5)
        assert results[0].input_tokens == 10
        assert results[0].output_tokens == 5

    def test_json_output_parsed(self, mock_gateway):
        mock_gateway.complete.return_value = GatewayResponse(
            content='{"key": "value"}',
            model="gpt-4o-mini",
            latency_ms=100.0,
        )
        runner = EvalRunner(gateway=mock_gateway)
        _, results = runner.run([_candidate()], [_case()])

        assert results[0].parsed_output == {"key": "value"}


class TestMultiTurnEval:
    """Tests for multi-turn conversation eval cases."""

    def _multi_turn_case(self, **kwargs) -> EvalCase:
        defaults = dict(
            id=str(uuid.uuid4()),
            input="What is the population?",
            expected_output=None,
            turns=[
                ConversationTurn(role="user", content="What is the capital of France?"),
                ConversationTurn(role="assistant", content="The capital of France is Paris."),
                ConversationTurn(role="user", content="What is the population?"),
            ],
        )
        defaults.update(kwargs)
        return EvalCase(**defaults)

    def test_multi_turn_case_is_detected(self):
        case = self._multi_turn_case()
        assert case.is_multi_turn is True

    def test_single_turn_case_not_multi_turn(self):
        case = _case()
        assert case.is_multi_turn is False

    def test_multi_turn_sends_messages(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        cand = _candidate(system_prompt="You are helpful.")
        case = self._multi_turn_case()
        runner.run([cand], [case])

        call_kwargs = mock_gateway.complete.call_args.kwargs
        assert call_kwargs["messages"] is not None
        msgs = call_kwargs["messages"]
        # system + 3 turns
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"

    def test_multi_turn_without_system_prompt(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        cand = _candidate(system_prompt=None)
        case = self._multi_turn_case()
        runner.run([cand], [case])

        call_kwargs = mock_gateway.complete.call_args.kwargs
        msgs = call_kwargs["messages"]
        # No system prompt, just 3 turns
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"

    def test_multi_turn_prompt_template_applied_to_user_turns(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        cand = _candidate(prompt_text="Please answer: {input}", system_prompt=None)
        case = self._multi_turn_case()
        runner.run([cand], [case])

        call_kwargs = mock_gateway.complete.call_args.kwargs
        msgs = call_kwargs["messages"]
        # User turns should have prompt template applied
        assert "Please answer: What is the capital of France?" in msgs[0]["content"]
        # Assistant turn should be unchanged
        assert msgs[1]["content"] == "The capital of France is Paris."
        assert "Please answer: What is the population?" in msgs[2]["content"]

    def test_multi_turn_dry_run(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway, dry_run=True)
        case = self._multi_turn_case()
        _, results = runner.run([_candidate()], [case])

        mock_gateway.complete.assert_not_called()
        assert "[DRY RUN" in results[0].raw_output

    def test_multi_turn_result_has_correct_ids(self, mock_gateway):
        runner = EvalRunner(gateway=mock_gateway)
        cand = _candidate()
        case = self._multi_turn_case()
        run_id, results = runner.run([cand], [case])

        assert results[0].candidate_id == cand.id
        assert results[0].eval_case_id == case.id
        assert results[0].run_id == run_id

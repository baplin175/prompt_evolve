"""Tests for multi-turn conversation eval case support."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.models.eval_case import ConversationTurn, EvalCase
from app.storage import db as storage


class TestConversationTurnModel:
    def test_create_turn(self):
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_turn_serialization(self):
        turn = ConversationTurn(role="assistant", content="Hi there!")
        data = turn.model_dump()
        assert data == {"role": "assistant", "content": "Hi there!"}


class TestEvalCaseMultiTurn:
    def test_single_turn_case_defaults(self):
        case = EvalCase(input="Hello")
        assert case.turns == []
        assert case.is_multi_turn is False

    def test_multi_turn_case(self):
        turns = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
            ConversationTurn(role="user", content="How are you?"),
        ]
        case = EvalCase(input="How are you?", turns=turns)
        assert case.is_multi_turn is True
        assert len(case.turns) == 3

    def test_multi_turn_from_dict(self):
        """Test creating a multi-turn case from raw dict (like JSONL loading)."""
        data = {
            "id": "test-mt",
            "input": "Follow-up question",
            "turns": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow-up question"},
            ],
        }
        case = EvalCase(**data)
        assert case.is_multi_turn is True
        assert len(case.turns) == 3
        assert case.turns[0].role == "user"
        assert case.turns[1].role == "assistant"

    def test_empty_turns_means_single_turn(self):
        case = EvalCase(input="test", turns=[])
        assert case.is_multi_turn is False

    def test_multi_turn_serialization_roundtrip(self):
        turns = [
            ConversationTurn(role="user", content="Q1"),
            ConversationTurn(role="assistant", content="A1"),
            ConversationTurn(role="user", content="Q2"),
        ]
        case = EvalCase(input="Q2", turns=turns, tags=["multi-turn"])
        data = case.model_dump()

        restored = EvalCase(**data)
        assert restored.is_multi_turn is True
        assert len(restored.turns) == 3
        assert restored.turns[0].content == "Q1"
        assert restored.turns[2].content == "Q2"


class TestMultiTurnStorage:
    def test_upsert_and_read_multi_turn(self, tmp_db):
        turns = [
            ConversationTurn(role="user", content="What is 2+2?"),
            ConversationTurn(role="assistant", content="4"),
            ConversationTurn(role="user", content="And 3+3?"),
        ]
        case = EvalCase(
            id="mt-store-001",
            input="And 3+3?",
            expected_output="6",
            turns=turns,
            tags=["multi-turn", "math"],
            difficulty="easy",
        )
        storage.upsert_eval_case(tmp_db, case)

        cases = storage.list_eval_cases(tmp_db)
        assert len(cases) == 1
        loaded = cases[0]
        assert loaded.id == "mt-store-001"
        assert loaded.is_multi_turn is True
        assert len(loaded.turns) == 3
        assert loaded.turns[0].role == "user"
        assert loaded.turns[0].content == "What is 2+2?"
        assert loaded.turns[1].role == "assistant"
        assert loaded.turns[2].content == "And 3+3?"

    def test_single_turn_backward_compat(self, tmp_db):
        case = EvalCase(
            id="st-store-001",
            input="What is the capital of France?",
            expected_output="Paris",
        )
        storage.upsert_eval_case(tmp_db, case)

        cases = storage.list_eval_cases(tmp_db)
        assert len(cases) == 1
        loaded = cases[0]
        assert loaded.is_multi_turn is False
        assert loaded.turns == []

    def test_jsonl_loading_multi_turn(self, tmp_path):
        """Test that multi-turn cases load correctly from JSONL files."""
        from app.storage.artifacts import load_eval_cases_from_jsonl

        jsonl_path = tmp_path / "test_eval.jsonl"
        records = [
            {
                "id": "jt-001",
                "input": "Follow up",
                "turns": [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Follow up"},
                ],
            },
            {
                "id": "jt-002",
                "input": "Simple question",
            },
        ]
        with jsonl_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        loaded = load_eval_cases_from_jsonl(jsonl_path)
        assert len(loaded) == 2

        mt_case = EvalCase(**loaded[0])
        assert mt_case.is_multi_turn is True
        assert len(mt_case.turns) == 3

        st_case = EvalCase(**loaded[1])
        assert st_case.is_multi_turn is False

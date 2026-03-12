"""Shared fixtures for tests."""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from app.models.candidate import PromptCandidate
from app.models.eval_case import EvalCase
from app.models.run_result import RunResult
from app.storage import db as storage


@pytest.fixture()
def tmp_db(tmp_path: Path):
    """Return a fresh, initialised SQLite connection in a temp directory."""
    db_path = tmp_path / "test.db"
    storage.init_db(db_path)
    conn = storage.get_connection(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def sample_candidate() -> PromptCandidate:
    return PromptCandidate(
        id=str(uuid.uuid4()),
        prompt_text="Answer the following question: {input}",
        system_prompt="You are a helpful assistant.",
        model="gpt-4o-mini",
        temperature=0.7,
        mutation_strategy="baseline",
    )


@pytest.fixture()
def sample_eval_case() -> EvalCase:
    return EvalCase(
        id=str(uuid.uuid4()),
        input="What is the capital of France?",
        expected_output="Paris",
        tags=["factual"],
        difficulty="easy",
    )


@pytest.fixture()
def sample_run_result(sample_candidate, sample_eval_case) -> RunResult:
    return RunResult(
        id=str(uuid.uuid4()),
        run_id="test-run-001",
        candidate_id=sample_candidate.id,
        eval_case_id=sample_eval_case.id,
        raw_output="Paris",
        latency_ms=250.0,
        input_tokens=20,
        output_tokens=5,
    )


@pytest.fixture()
def mock_gateway(mocker):
    """A mock gateway that returns a preset response."""
    from app.gateway.base import GatewayResponse

    gw = mocker.MagicMock()
    gw.complete.return_value = GatewayResponse(
        content="Mocked LLM response",
        model="gpt-4o-mini",
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
    )
    return gw

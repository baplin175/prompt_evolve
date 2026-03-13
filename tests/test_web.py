"""Tests for the Flask web UI."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.models.candidate import PromptCandidate
from app.models.eval_case import ConversationTurn, EvalCase
from app.models.run_result import RunResult
from app.models.score import CandidateScore
from app.storage import db as storage
from app.web import create_app


@pytest.fixture()
def web_db(tmp_path: Path):
    """Return a fresh, initialised DB path for the web app."""
    db_path = tmp_path / "web_test.db"
    storage.init_db(db_path)
    return db_path


@pytest.fixture()
def client(web_db):
    """Flask test client backed by an empty database."""
    app = create_app(db_path=web_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture()
def seeded_client(web_db):
    """Flask test client backed by a database with sample data."""
    conn = storage.get_connection(web_db)

    # Insert a candidate
    candidate = PromptCandidate(
        id="cand-0001",
        prompt_text="Answer the following question: {input}",
        system_prompt="You are a helpful assistant.",
        model="gpt-4o-mini",
        temperature=0.7,
        mutation_strategy="baseline",
    )
    storage.upsert_candidate(conn, candidate)

    # Insert a child candidate
    child = PromptCandidate(
        id="cand-0002",
        parent_id="cand-0001",
        prompt_text="Please answer: {input}",
        model="gpt-4o-mini",
        temperature=0.5,
        mutation_strategy="simplify_wording",
    )
    storage.upsert_candidate(conn, child)

    # Insert eval case
    eval_case = EvalCase(
        id="eval-0001",
        input="What is the capital of France?",
        expected_output="Paris",
        tags=["factual"],
        difficulty="easy",
    )
    storage.upsert_eval_case(conn, eval_case)

    # Insert run result
    run_result = RunResult(
        id="rr-0001",
        run_id="run-0001",
        candidate_id="cand-0001",
        eval_case_id="eval-0001",
        raw_output="Paris",
        latency_ms=250.0,
        input_tokens=20,
        output_tokens=5,
    )
    storage.insert_run_result(conn, run_result)

    # Insert candidate score
    cs = CandidateScore(
        candidate_id="cand-0001",
        run_id="run-0001",
        aggregate_score=0.85,
        dimension_scores={"correctness": 0.9, "format_compliance": 0.8},
        eval_case_count=1,
        error_count=0,
        avg_latency_ms=250.0,
        total_cost_usd=0.001,
    )
    storage.upsert_candidate_score(conn, cs)

    conn.close()

    app = create_app(db_path=web_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestDashboard:
    def test_dashboard_empty(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data

    def test_dashboard_with_data(self, seeded_client):
        resp = seeded_client.get("/")
        assert resp.status_code == 200
        assert b"cand-0001" in resp.data


class TestCandidates:
    def test_candidates_list_empty(self, client):
        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert b"No candidates found" in resp.data

    def test_candidates_list_with_data(self, seeded_client):
        resp = seeded_client.get("/candidates")
        assert resp.status_code == 200
        assert b"cand-0001" in resp.data
        assert b"baseline" in resp.data

    def test_candidate_detail(self, seeded_client):
        resp = seeded_client.get("/candidates/cand-0001")
        assert resp.status_code == 200
        assert b"Answer the following question" in resp.data
        assert b"baseline" in resp.data

    def test_candidate_detail_shows_children(self, seeded_client):
        resp = seeded_client.get("/candidates/cand-0001")
        assert resp.status_code == 200
        assert b"cand-0002" in resp.data

    def test_candidate_detail_shows_scores(self, seeded_client):
        resp = seeded_client.get("/candidates/cand-0001")
        assert resp.status_code == 200
        assert b"0.8500" in resp.data

    def test_candidate_not_found(self, seeded_client):
        resp = seeded_client.get("/candidates/nonexistent")
        assert resp.status_code == 404


class TestRuns:
    def test_runs_list_empty(self, client):
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert b"No runs found" in resp.data

    def test_runs_list_with_data(self, seeded_client):
        resp = seeded_client.get("/runs")
        assert resp.status_code == 200
        assert b"run-0001" in resp.data

    def test_run_detail(self, seeded_client):
        resp = seeded_client.get("/runs/run-0001")
        assert resp.status_code == 200
        assert b"run-0001" in resp.data
        assert b"Paris" in resp.data

    def test_run_detail_shows_leaderboard(self, seeded_client):
        resp = seeded_client.get("/runs/run-0001")
        assert resp.status_code == 200
        assert b"Leaderboard" in resp.data

    def test_run_not_found(self, seeded_client):
        resp = seeded_client.get("/runs/nonexistent")
        assert resp.status_code == 404


class TestEvalCases:
    def test_eval_cases_empty(self, client):
        resp = client.get("/eval-cases")
        assert resp.status_code == 200
        assert b"No eval cases found" in resp.data

    def test_eval_cases_with_data(self, seeded_client):
        resp = seeded_client.get("/eval-cases")
        assert resp.status_code == 200
        assert b"capital of France" in resp.data
        assert b"easy" in resp.data

    def test_eval_cases_shows_single_turn_badge(self, seeded_client):
        resp = seeded_client.get("/eval-cases")
        assert resp.status_code == 200
        assert b"single-turn" in resp.data

    def test_eval_cases_shows_multi_turn_badge(self, web_db):
        conn = storage.get_connection(web_db)
        multi_case = EvalCase(
            id="eval-mt-001",
            input="What is the population?",
            expected_output=None,
            tags=["multi-turn"],
            difficulty="medium",
            turns=[
                ConversationTurn(role="user", content="What is the capital of France?"),
                ConversationTurn(role="assistant", content="Paris."),
                ConversationTurn(role="user", content="What is the population?"),
            ],
        )
        storage.upsert_eval_case(conn, multi_case)
        conn.close()

        app = create_app(db_path=web_db)
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.get("/eval-cases")
            assert resp.status_code == 200
            assert b"multi-turn" in resp.data


class TestRounds:
    def test_rounds_empty(self, client):
        resp = client.get("/rounds")
        assert resp.status_code == 200
        assert b"No optimization rounds found" in resp.data

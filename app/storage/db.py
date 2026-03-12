"""SQLite storage backend — schema definition and helper functions."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.models.candidate import PromptCandidate
from app.models.eval_case import EvalCase
from app.models.round_summary import OptimizationRound
from app.models.run_result import RunResult
from app.models.score import CandidateScore, ScoreBreakdown

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    prompt_text TEXT NOT NULL,
    system_prompt TEXT,
    model TEXT NOT NULL,
    temperature REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    mutation_strategy TEXT NOT NULL DEFAULT 'baseline',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id TEXT PRIMARY KEY,
    input TEXT NOT NULL,
    expected_output TEXT,
    reference_output TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    difficulty TEXT NOT NULL DEFAULT 'medium',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    eval_case_id TEXT NOT NULL,
    raw_output TEXT NOT NULL DEFAULT '',
    parsed_output TEXT,
    error TEXT,
    latency_ms REAL NOT NULL DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS score_breakdowns (
    id TEXT PRIMARY KEY,
    run_result_id TEXT NOT NULL,
    scorer_name TEXT NOT NULL,
    dimension TEXT NOT NULL,
    raw_score REAL NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    weighted_score REAL NOT NULL DEFAULT 0.0,
    details TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS candidate_scores (
    candidate_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    aggregate_score REAL NOT NULL,
    dimension_scores TEXT NOT NULL DEFAULT '{}',
    eval_case_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL NOT NULL DEFAULT 0,
    total_cost_usd REAL,
    PRIMARY KEY (candidate_id, run_id)
);

CREATE TABLE IF NOT EXISTS optimization_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidates_evaluated TEXT NOT NULL DEFAULT '[]',
    top_candidate_ids TEXT NOT NULL DEFAULT '[]',
    scores TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set to Row."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Create all tables if they don't already exist."""
    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA)
    logger.info("Database initialised at %s", db_path)


# ---------------------------------------------------------------------------
# Candidate helpers
# ---------------------------------------------------------------------------


def upsert_candidate(conn: sqlite3.Connection, candidate: PromptCandidate) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO candidates
           (id, parent_id, prompt_text, system_prompt, model, temperature,
            metadata, mutation_strategy, notes, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            candidate.id,
            candidate.parent_id,
            candidate.prompt_text,
            candidate.system_prompt,
            candidate.model,
            candidate.temperature,
            json.dumps(candidate.metadata),
            candidate.mutation_strategy,
            candidate.notes,
            candidate.created_at,
        ),
    )
    conn.commit()


def get_candidate(conn: sqlite3.Connection, candidate_id: str) -> Optional[PromptCandidate]:
    row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if row is None:
        return None
    return _row_to_candidate(row)


def list_candidates(conn: sqlite3.Connection) -> list[PromptCandidate]:
    rows = conn.execute("SELECT * FROM candidates ORDER BY created_at").fetchall()
    return [_row_to_candidate(r) for r in rows]


def _row_to_candidate(row: sqlite3.Row) -> PromptCandidate:
    return PromptCandidate(
        id=row["id"],
        parent_id=row["parent_id"],
        prompt_text=row["prompt_text"],
        system_prompt=row["system_prompt"],
        model=row["model"],
        temperature=row["temperature"],
        metadata=json.loads(row["metadata"]),
        mutation_strategy=row["mutation_strategy"],
        notes=row["notes"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# EvalCase helpers
# ---------------------------------------------------------------------------


def upsert_eval_case(conn: sqlite3.Connection, case: EvalCase) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO eval_cases
           (id, input, expected_output, reference_output, tags, difficulty, metadata)
           VALUES (?,?,?,?,?,?,?)""",
        (
            case.id,
            case.input,
            case.expected_output,
            case.reference_output,
            json.dumps(case.tags),
            case.difficulty,
            json.dumps(case.metadata),
        ),
    )
    conn.commit()


def list_eval_cases(conn: sqlite3.Connection) -> list[EvalCase]:
    rows = conn.execute("SELECT * FROM eval_cases").fetchall()
    return [_row_to_eval_case(r) for r in rows]


def _row_to_eval_case(row: sqlite3.Row) -> EvalCase:
    return EvalCase(
        id=row["id"],
        input=row["input"],
        expected_output=row["expected_output"],
        reference_output=row["reference_output"],
        tags=json.loads(row["tags"]),
        difficulty=row["difficulty"],
        metadata=json.loads(row["metadata"]),
    )


# ---------------------------------------------------------------------------
# RunResult helpers
# ---------------------------------------------------------------------------


def insert_run_result(conn: sqlite3.Connection, result: RunResult) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO run_results
           (id, run_id, candidate_id, eval_case_id, raw_output, parsed_output,
            error, latency_ms, input_tokens, output_tokens, cost_usd, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result.id,
            result.run_id,
            result.candidate_id,
            result.eval_case_id,
            result.raw_output,
            json.dumps(result.parsed_output) if result.parsed_output is not None else None,
            result.error,
            result.latency_ms,
            result.input_tokens,
            result.output_tokens,
            result.cost_usd,
            result.created_at,
        ),
    )
    conn.commit()


def get_run_results(conn: sqlite3.Connection, run_id: str) -> list[RunResult]:
    rows = conn.execute(
        "SELECT * FROM run_results WHERE run_id = ?", (run_id,)
    ).fetchall()
    return [_row_to_run_result(r) for r in rows]


def get_run_results_for_candidate(
    conn: sqlite3.Connection, run_id: str, candidate_id: str
) -> list[RunResult]:
    rows = conn.execute(
        "SELECT * FROM run_results WHERE run_id = ? AND candidate_id = ?",
        (run_id, candidate_id),
    ).fetchall()
    return [_row_to_run_result(r) for r in rows]


def _row_to_run_result(row: sqlite3.Row) -> RunResult:
    parsed = row["parsed_output"]
    return RunResult(
        id=row["id"],
        run_id=row["run_id"],
        candidate_id=row["candidate_id"],
        eval_case_id=row["eval_case_id"],
        raw_output=row["raw_output"],
        parsed_output=json.loads(parsed) if parsed else None,
        error=row["error"],
        latency_ms=row["latency_ms"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def insert_score_breakdown(conn: sqlite3.Connection, bd: ScoreBreakdown) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO score_breakdowns
           (id, run_result_id, scorer_name, dimension, raw_score, weight, weighted_score, details)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            bd.id,
            bd.run_result_id,
            bd.scorer_name,
            bd.dimension,
            bd.raw_score,
            bd.weight,
            bd.weighted_score,
            json.dumps(bd.details),
        ),
    )
    conn.commit()


def upsert_candidate_score(conn: sqlite3.Connection, cs: CandidateScore) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO candidate_scores
           (candidate_id, run_id, aggregate_score, dimension_scores, eval_case_count,
            error_count, avg_latency_ms, total_cost_usd)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            cs.candidate_id,
            cs.run_id,
            cs.aggregate_score,
            json.dumps(cs.dimension_scores),
            cs.eval_case_count,
            cs.error_count,
            cs.avg_latency_ms,
            cs.total_cost_usd,
        ),
    )
    conn.commit()


def get_candidate_scores(conn: sqlite3.Connection, run_id: str) -> list[CandidateScore]:
    rows = conn.execute(
        "SELECT * FROM candidate_scores WHERE run_id = ? ORDER BY aggregate_score DESC",
        (run_id,),
    ).fetchall()
    return [
        CandidateScore(
            candidate_id=r["candidate_id"],
            run_id=r["run_id"],
            aggregate_score=r["aggregate_score"],
            dimension_scores=json.loads(r["dimension_scores"]),
            eval_case_count=r["eval_case_count"],
            error_count=r["error_count"],
            avg_latency_ms=r["avg_latency_ms"],
            total_cost_usd=r["total_cost_usd"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Round summary helpers
# ---------------------------------------------------------------------------


def insert_optimization_round(conn: sqlite3.Connection, rnd: OptimizationRound) -> None:
    conn.execute(
        """INSERT INTO optimization_rounds
           (round_number, strategy, run_id, candidates_evaluated, top_candidate_ids,
            scores, metadata, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            rnd.round_number,
            rnd.strategy,
            rnd.run_id,
            json.dumps(rnd.candidates_evaluated),
            json.dumps(rnd.top_candidate_ids),
            json.dumps(rnd.scores),
            json.dumps(rnd.metadata),
            rnd.created_at,
        ),
    )
    conn.commit()


def list_optimization_rounds(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM optimization_rounds ORDER BY round_number"
    ).fetchall()
    return [dict(r) for r in rows]

"""Flask web UI for interactive exploration of prompt evolution data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flask import Flask, render_template, abort

from app import config as cfg
from app.models.score import CandidateScore
from app.reporting.leaderboard import build_leaderboard
from app.storage import db as storage

logger = logging.getLogger(__name__)


def create_app(db_path: Path | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        db_path: Path to the SQLite database.  Falls back to ``cfg.DB_PATH``.

    Returns:
        Configured Flask application instance.
    """
    resolved_db = db_path or cfg.DB_PATH

    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

    app.config["DB_PATH"] = resolved_db

    def _conn():
        return storage.get_connection(app.config["DB_PATH"])

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def dashboard():
        """Dashboard showing summary stats and recent activity."""
        conn = _conn()
        try:
            candidates = storage.list_candidates(conn)
            eval_cases = storage.list_eval_cases(conn)
            rounds = storage.list_optimization_rounds(conn)

            # Collect distinct run_ids from candidate_scores
            run_rows = conn.execute(
                "SELECT DISTINCT run_id FROM candidate_scores ORDER BY run_id"
            ).fetchall()
            run_ids = [r["run_id"] for r in run_rows]

            return render_template(
                "dashboard.html",
                candidates=candidates,
                eval_cases=eval_cases,
                rounds=rounds,
                run_ids=run_ids,
            )
        finally:
            conn.close()

    @app.route("/candidates")
    def candidates_list():
        """List all prompt candidates."""
        conn = _conn()
        try:
            candidates = storage.list_candidates(conn)
            return render_template("candidates.html", candidates=candidates)
        finally:
            conn.close()

    @app.route("/candidates/<candidate_id>")
    def candidate_detail(candidate_id: str):
        """Detail view of a single candidate."""
        conn = _conn()
        try:
            candidate = storage.get_candidate(conn, candidate_id)
            if candidate is None:
                abort(404)

            # Fetch scores for this candidate across all runs
            rows = conn.execute(
                "SELECT * FROM candidate_scores WHERE candidate_id = ? "
                "ORDER BY aggregate_score DESC",
                (candidate_id,),
            ).fetchall()
            scores = [
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

            # Find children of this candidate
            children = [
                c
                for c in storage.list_candidates(conn)
                if c.parent_id == candidate_id
            ]

            return render_template(
                "candidate_detail.html",
                candidate=candidate,
                scores=scores,
                children=children,
            )
        finally:
            conn.close()

    @app.route("/runs")
    def runs_list():
        """List all evaluation runs."""
        conn = _conn()
        try:
            run_rows = conn.execute(
                "SELECT run_id, COUNT(*) as result_count, "
                "SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) as error_count, "
                "ROUND(AVG(latency_ms), 1) as avg_latency_ms, "
                "MIN(created_at) as started_at "
                "FROM run_results GROUP BY run_id ORDER BY started_at DESC"
            ).fetchall()
            runs = [dict(r) for r in run_rows]
            return render_template("runs.html", runs=runs)
        finally:
            conn.close()

    @app.route("/runs/<run_id>")
    def run_detail(run_id: str):
        """Detail view of a single run with leaderboard and results."""
        conn = _conn()
        try:
            results = storage.get_run_results(conn, run_id)
            if not results:
                abort(404)

            scores = storage.get_candidate_scores(conn, run_id)
            candidates = storage.list_candidates(conn)
            leaderboard = build_leaderboard(scores, candidates) if scores else []

            return render_template(
                "run_detail.html",
                run_id=run_id,
                results=results,
                scores=scores,
                leaderboard=leaderboard,
            )
        finally:
            conn.close()

    @app.route("/eval-cases")
    def eval_cases_list():
        """List all evaluation cases."""
        conn = _conn()
        try:
            cases = storage.list_eval_cases(conn)
            return render_template("eval_cases.html", eval_cases=cases)
        finally:
            conn.close()

    @app.route("/rounds")
    def rounds_list():
        """List all optimization rounds."""
        conn = _conn()
        try:
            rounds = storage.list_optimization_rounds(conn)
            # Parse JSON fields for display
            for rnd in rounds:
                for key in ("candidates_evaluated", "top_candidate_ids", "scores", "metadata"):
                    if isinstance(rnd.get(key), str):
                        try:
                            rnd[key] = json.loads(rnd[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
            return render_template("rounds.html", rounds=rounds)
        finally:
            conn.close()

    return app

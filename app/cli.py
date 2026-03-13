"""Click-based CLI for the Prompt Evolution Rig."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import click

from app import config as cfg
from app.models.candidate import PromptCandidate
from app.models.eval_case import EvalCase
from app.storage import db as storage
from app.storage.artifacts import load_eval_cases_from_jsonl

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prompt-evolve")


def _get_db_path(db_path: Optional[str]) -> Path:
    return Path(db_path) if db_path else cfg.DB_PATH


def _get_conn(db_path: Path):
    return storage.get_connection(db_path)


def _load_gateway():
    if cfg.LLM_GATEWAY == "matcha":
        from app.gateway.matcha_client import MatchaClient
        return MatchaClient()
    from app.gateway.openai_client import OpenAIClient
    return OpenAIClient()


def _load_eval_cases(eval_set: str, conn) -> list[EvalCase]:
    """Load eval cases from JSONL and upsert into DB."""
    raw_cases = load_eval_cases_from_jsonl(Path(eval_set))
    cases = []
    for raw in raw_cases:
        case = EvalCase(**raw)
        storage.upsert_eval_case(conn, case)
        cases.append(case)
    return cases


@click.group()
@click.option("--db", default=None, help="Path to SQLite database file")
@click.pass_context
def cli(ctx, db):
    """Prompt Evolution Rig — evolve prompts with eval-driven search."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the database and data directories."""
    db_path = _get_db_path(ctx.obj.get("db"))
    storage.init_db(db_path)

    for d in ["data/eval_cases", "data/prompts", "data/runs", "data/reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    click.echo(f"✓ Database initialised at {db_path}")
    click.echo("✓ Data directories created: data/eval_cases, data/prompts, data/runs, data/reports")


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


@cli.group()
def prompt():
    """Manage prompt candidates."""


@prompt.command("add")
@click.option("--file", "-f", required=True, help="Path to prompt JSON file")
@click.pass_context
def prompt_add(ctx, file):
    """Add a prompt candidate from a JSON file."""
    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    data = json.loads(Path(file).read_text(encoding="utf-8"))
    candidate = PromptCandidate(**data)
    storage.upsert_candidate(conn, candidate)
    click.echo(f"✓ Prompt candidate added: {candidate.id}")
    click.echo(f"  Strategy: {candidate.mutation_strategy}")
    click.echo(f"  Model: {candidate.model} (temp={candidate.temperature})")


@prompt.command("list")
@click.pass_context
def prompt_list(ctx):
    """List all prompt candidates."""
    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    candidates = storage.list_candidates(conn)
    if not candidates:
        click.echo("No prompt candidates found. Use 'prompt-evolve prompt add --file <path>'")
        return

    click.echo(f"{'ID':38} {'Strategy':28} {'Model':15} {'Temp':6} {'Created'}")
    click.echo("-" * 110)
    for c in candidates:
        click.echo(
            f"{c.id:38} {c.mutation_strategy:28} {c.model:15} {c.temperature:<6} {c.created_at[:19]}"
        )


# ---------------------------------------------------------------------------
# mutate
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--candidate-id", required=True, help="Source candidate UUID")
@click.option("--operators", required=True, help="Comma-separated mutation operators")
@click.option("--n", default=1, show_default=True, help="Number of variants to generate")
@click.pass_context
def mutate(ctx, candidate_id, operators, n):
    """Run mutations on a prompt candidate."""
    from app.mutations.engine import MutationEngine

    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    candidate = storage.get_candidate(conn, candidate_id)
    if candidate is None:
        click.echo(f"Error: candidate {candidate_id} not found", err=True)
        sys.exit(1)

    gateway = _load_gateway()
    engine = MutationEngine(gateway)
    ops = [o.strip() for o in operators.split(",")]

    click.echo(f"Mutating candidate {candidate_id[:8]} with operators: {ops} (n={n})")
    children = engine.mutate(candidate, ops, n=n)

    for child in children:
        storage.upsert_candidate(conn, child)
        click.echo(f"  ✓ New candidate: {child.id} (strategy={child.mutation_strategy})")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command("run")
@click.option("--eval-set", required=True, help="Path to eval cases JSONL file")
@click.option("--candidate-id", default=None, help="Specific candidate ID to evaluate")
@click.option("--run-id", default=None, help="Optional run ID")
@click.option("--dry-run", is_flag=True, help="Skip LLM calls; use placeholder outputs")
@click.option("--output-dir", default="data/runs", show_default=True, help="Directory for run artifacts")
@click.option("--max-tokens", default=1024, show_default=True)
@click.pass_context
def run_eval(ctx, eval_set, candidate_id, run_id, dry_run, output_dir, max_tokens):
    """Evaluate prompt candidates on an eval set."""
    from app.runner.runner import EvalRunner

    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    eval_cases = _load_eval_cases(eval_set, conn)
    click.echo(f"Loaded {len(eval_cases)} eval cases from {eval_set}")

    if candidate_id:
        candidate = storage.get_candidate(conn, candidate_id)
        if candidate is None:
            click.echo(f"Error: candidate {candidate_id} not found", err=True)
            sys.exit(1)
        candidates = [candidate]
    else:
        candidates = storage.list_candidates(conn)
        if not candidates:
            click.echo("No candidates found. Add one with 'prompt-evolve prompt add'", err=True)
            sys.exit(1)

    click.echo(f"Evaluating {len(candidates)} candidate(s) on {len(eval_cases)} eval case(s)")

    gateway = _load_gateway() if not dry_run else _load_gateway()
    runner = EvalRunner(
        gateway=gateway,
        max_tokens=max_tokens,
        run_dir=Path(output_dir),
        dry_run=dry_run,
    )

    actual_run_id, results = runner.run(candidates, eval_cases, run_id=run_id)

    for result in results:
        storage.insert_run_result(conn, result)

    errors = sum(1 for r in results if r.error)
    click.echo(
        f"✓ Run complete. ID={actual_run_id} | Results={len(results)} | Errors={errors}"
    )
    click.echo(f"  Artifacts: data/runs/{actual_run_id}/results.jsonl")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--run-id", required=True, help="Run ID to score")
@click.option("--config", "config_path", default="data/scoring_config.json",
              show_default=True, help="Scoring config JSON file")
@click.pass_context
def score(ctx, run_id, config_path):
    """Score all results for a run."""
    from app.scoring.aggregator import aggregate_candidate_scores
    from app.scoring.deterministic import ALL_DETERMINISTIC_SCORERS

    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    scoring_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    results = storage.get_run_results(conn, run_id)
    if not results:
        click.echo(f"No results found for run {run_id}", err=True)
        sys.exit(1)

    cases = storage.list_eval_cases(conn)
    candidates = storage.list_candidates(conn)
    case_map = {c.id: c for c in cases}

    scorers_config = scoring_config.get("scorers", [])
    candidate_ids = list({r.candidate_id for r in results})

    for cid in candidate_ids:
        cand_results = [r for r in results if r.candidate_id == cid]
        all_breakdowns = []

        for result in cand_results:
            case = case_map.get(result.eval_case_id)
            expected = case.expected_output if case else None

            for scorer_cfg in scorers_config:
                scorer_name = scorer_cfg.get("name")
                if scorer_name == "model_judge":
                    continue  # Skip for CLI score command without explicit flag
                scorer = ALL_DETERMINISTIC_SCORERS.get(scorer_name)
                if scorer is None:
                    continue
                try:
                    bd = scorer.score(result, expected, scorer_cfg)
                    all_breakdowns.append(bd)
                    storage.insert_score_breakdown(conn, bd)
                except Exception as exc:
                    logger.error("Scorer '%s' error: %s", scorer_name, exc)

        cs = aggregate_candidate_scores(
            candidate_id=cid,
            run_id=run_id,
            results=cand_results,
            breakdowns=all_breakdowns,
            scoring_config=scoring_config,
        )
        storage.upsert_candidate_score(conn, cs)
        click.echo(f"  Candidate {cid[:8]}: aggregate={cs.aggregate_score:.4f}")

    click.echo(f"✓ Scoring complete for run {run_id}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--run-id", required=True, help="Run ID to report on")
@click.option("--output", default=None, help="Output path for markdown report")
@click.pass_context
def report(ctx, run_id, output):
    """Generate a markdown report for a scored run."""
    from app.reporting.report import generate_report

    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    scores = storage.get_candidate_scores(conn, run_id)
    if not scores:
        click.echo(f"No scores found for run {run_id}. Run 'prompt-evolve score --run-id {run_id}' first.", err=True)
        sys.exit(1)

    results = storage.get_run_results(conn, run_id)
    candidates = storage.list_candidates(conn)
    eval_cases = storage.list_eval_cases(conn)
    rounds = []

    output_path = Path(output) if output else Path(f"data/reports/report_{run_id[:8]}.md")

    content = generate_report(
        run_id=run_id,
        scores=scores,
        candidates=candidates,
        eval_cases=eval_cases,
        results=results,
        rounds=rounds,
        strategy="manual",
        output_path=output_path,
    )

    click.echo(f"✓ Report written to {output_path}")
    click.echo(f"  Best candidate score: {scores[0].aggregate_score:.4f}")


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--baseline", required=True, help="Path to baseline prompt JSON")
@click.option("--eval-set", required=True, help="Path to eval cases JSONL file")
@click.option("--strategy", default="hill_climbing",
              type=click.Choice(["hill_climbing", "beam_search"]), show_default=True)
@click.option("--rounds", default=3, show_default=True, help="Number of evolution rounds")
@click.option("--beam-width", default=3, show_default=True, help="Beam width (beam_search only)")
@click.option("--variants-per-candidate", default=2, show_default=True)
@click.option("--operators", default="simplify_wording,tighten_constraints",
              show_default=True, help="Comma-separated mutation operators")
@click.option("--output-dir", default="data/runs", show_default=True)
@click.option("--scoring-config", "scoring_config_path",
              default="data/scoring_config.json", show_default=True)
@click.option("--report-output", default=None, help="Path for final report")
@click.option("--max-tokens", default=1024, show_default=True)
@click.option("--dry-run", is_flag=True, help="Skip LLM calls")
@click.pass_context
def optimize(
    ctx,
    baseline,
    eval_set,
    strategy,
    rounds,
    beam_width,
    variants_per_candidate,
    operators,
    output_dir,
    scoring_config_path,
    report_output,
    max_tokens,
    dry_run,
):
    """Run the full prompt evolution optimization loop."""
    from app.optimization.loop import EvolutionLoop
    from app.reporting.report import generate_report

    db_path = _get_db_path(ctx.obj.get("db"))
    conn = _get_conn(db_path)

    # Load baseline
    baseline_data = json.loads(Path(baseline).read_text(encoding="utf-8"))
    baseline_candidate = PromptCandidate(**baseline_data)
    storage.upsert_candidate(conn, baseline_candidate)

    # Load eval cases
    eval_cases = _load_eval_cases(eval_set, conn)
    click.echo(f"Loaded {len(eval_cases)} eval cases")

    # Load scoring config
    scoring_config = json.loads(Path(scoring_config_path).read_text(encoding="utf-8"))

    mutation_ops = [o.strip() for o in operators.split(",")]

    gateway = _load_gateway()

    loop = EvolutionLoop(
        gateway=gateway,
        eval_cases=eval_cases,
        scoring_config=scoring_config,
        mutation_operators=mutation_ops,
        strategy=strategy,
        rounds=rounds,
        variants_per_candidate=variants_per_candidate,
        beam_width=beam_width,
        max_tokens=max_tokens,
        output_dir=Path(output_dir),
        db_path=db_path,
        dry_run=dry_run,
    )

    best_candidate, all_rounds = loop.run(baseline_candidate)

    click.echo(f"\n✓ Optimization complete!")
    click.echo(f"  Best candidate: {best_candidate.id}")
    click.echo(f"  Strategy: {best_candidate.mutation_strategy}")

    # Generate final report
    if all_rounds:
        last_run_id = all_rounds[-1].run_id
        scores = storage.get_candidate_scores(conn, last_run_id)
        results = storage.get_run_results(conn, last_run_id)
        all_candidates = storage.list_candidates(conn)
        all_eval_cases = storage.list_eval_cases(conn)

        report_path = Path(report_output) if report_output else Path("data/reports/final_report.md")
        generate_report(
            run_id=last_run_id,
            scores=scores,
            candidates=all_candidates,
            eval_cases=all_eval_cases,
            results=results,
            rounds=all_rounds,
            strategy=strategy,
            output_path=report_path,
            extra_metadata={
                "Rounds": rounds,
                "Strategy": strategy,
                "Eval Set": eval_set,
                "Variants/Candidate": variants_per_candidate,
                "Best Candidate": best_candidate.id,
            },
        )
        click.echo(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option("--port", default=5000, show_default=True, help="Port to listen on")
@click.pass_context
def web(ctx, host, port):
    """Launch the web UI for interactive exploration."""
    from app.web import create_app

    db_path = _get_db_path(ctx.obj.get("db"))
    storage.init_db(db_path)

    app = create_app(db_path=db_path)
    click.echo(f"Starting web UI at http://{host}:{port}")
    click.echo(f"Database: {db_path}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    cli()

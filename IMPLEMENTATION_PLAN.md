# Implementation Plan: Prompt Evolution Rig

## Repository Assessment

The repository (`baplin175/prompt_evolve`) starts with only a stub `README.md`. Everything
must be built from scratch. The goal is a robust, end-to-end Prompt Evolution Rig that
evolves prompts toward the best-performing prompt for a specific outcome using eval-driven
search over prompt variants.

---

## Chosen Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | Modern type hints, rich ecosystem, required |
| Models / Validation | Pydantic v2 | Schema enforcement, JSON serialisation |
| CLI | Click | Ergonomic, composable, well-documented |
| Storage | SQLite (built-in `sqlite3`) | Zero deps, reproducible, queryable |
| LLM Gateway | OpenAI Python SDK | Works with any OpenAI-compatible endpoint |
| Config | `python-dotenv` + env vars | Portable, secrets-safe |
| Testing | pytest | Standard, composable |
| Run Artifacts | JSONL | Appendable, grep-able, schema-flexible |

---

## Proposed Module Structure

```
app/
  cli.py                    # Click-based CLI entrypoints
  config.py                 # Settings from env vars
  models/
    __init__.py
    candidate.py            # PromptCandidate
    eval_case.py            # EvalCase
    run_result.py           # RunResult
    score.py                # Score / ScoreBreakdown
    round_summary.py        # OptimizationRound
  gateway/
    __init__.py
    base.py                 # Abstract GatewayClient
    openai_client.py        # OpenAI-compatible implementation
  mutations/
    __init__.py
    base.py                 # MutationOperator protocol
    operators.py            # All 8 mutation operator implementations
    engine.py               # MutationEngine orchestrator
  runner/
    __init__.py
    runner.py               # EvalRunner
  scoring/
    __init__.py
    base.py                 # Scorer protocol
    deterministic.py        # Exact match, regex, JSON, length, custom
    judge.py                # ModelJudgeScorer
    aggregator.py           # Weighted score aggregation
  optimization/
    __init__.py
    hill_climbing.py        # HillClimbingSelector
    beam_search.py          # BeamSearchSelector
    loop.py                 # EvolutionLoop orchestrator
  reporting/
    __init__.py
    leaderboard.py          # Leaderboard builder
    report.py               # Markdown report generator
  storage/
    __init__.py
    db.py                   # SQLite schema + helpers
    artifacts.py            # JSONL artifact writer/reader
data/
  eval_cases/
    sample_eval.jsonl       # ≥10 diverse eval cases
  prompts/
    baseline.json           # Sample baseline prompt
  runs/                     # (runtime)
  reports/                  # (runtime)
  scoring_config.json       # Sample scoring config
tests/
  conftest.py
  test_scoring.py
  test_selection.py
  test_mutations.py
  test_runner.py
pyproject.toml
.env.example
README.md
IMPLEMENTATION_PLAN.md
```

---

## Data Model Design

### PromptCandidate
- `id`: UUID (auto-generated)
- `parent_id`: nullable UUID
- `prompt_text`: str (user-facing prompt, may contain `{input}` placeholder)
- `system_prompt`: Optional[str]
- `model`: str (default `gpt-4o-mini`)
- `temperature`: float (0.0–2.0)
- `metadata`: dict
- `mutation_strategy`: str (which operator created this)
- `notes`: str
- `created_at`: ISO timestamp

### EvalCase
- `id`: UUID
- `input`: str
- `expected_output`: Optional[str]
- `reference_output`: Optional[str]
- `tags`: List[str]
- `difficulty`: str (`easy`, `medium`, `hard`)
- `metadata`: dict

### RunResult
- `id`: UUID
- `run_id`: str
- `candidate_id`: UUID
- `eval_case_id`: UUID
- `raw_output`: str
- `parsed_output`: Optional[Any]
- `error`: Optional[str]
- `latency_ms`: float
- `input_tokens`: Optional[int]
- `output_tokens`: Optional[int]
- `cost_usd`: Optional[float]
- `created_at`: ISO timestamp

### ScoreBreakdown
- `run_result_id`: UUID
- `scorer_name`: str
- `dimension`: str
- `raw_score`: float (0–1)
- `weight`: float
- `weighted_score`: float
- `details`: dict

### OptimizationRound
- `round_number`: int
- `strategy`: str
- `candidates_evaluated`: List[UUID]
- `top_candidate_ids`: List[UUID]
- `scores`: dict
- `created_at`: ISO timestamp

---

## Optimization Loop Design

```
baseline prompt
  → generate N variants via mutation engine
  → evaluate all on eval set
  → score and rank
  → keep top K (hill climbing: K=1, beam: K>1)
  → generate next generation from top K
  → repeat for configurable rounds
  → save all artifacts
  → produce final report
```

The loop is deterministic given the same random seed, mutation operators, and eval set.

---

## Scoring Design

Scorers produce a float in [0.0, 1.0]. Dimensions:

| Dimension | Default Weight | Scorers |
|---|---|---|
| `correctness` | 0.50 | exact_match, substring_match, regex_match, model_judge |
| `robustness` | 0.20 | Pass rate across difficulty levels |
| `format_compliance` | 0.20 | json_field_presence, length_constraint, regex_match |
| `latency_efficiency` | 0.05 | Inverse-normalised latency |
| `cost_efficiency` | 0.05 | Inverse-normalised token cost |

Final candidate score = Σ (dimension_score × dimension_weight).

---

## CLI Design

```bash
prompt-evolve init
prompt-evolve prompt add --file data/prompts/baseline.json
prompt-evolve prompt list
prompt-evolve mutate --candidate-id <uuid> --operators simplify_wording,add_examples --n 3
prompt-evolve run --eval-set data/eval_cases/sample_eval.jsonl [--candidate-id <uuid>] [--dry-run]
prompt-evolve score --run-id <run_id> --config data/scoring_config.json
prompt-evolve report --run-id <run_id> --output data/reports/report.md
prompt-evolve optimize \
  --baseline data/prompts/baseline.json \
  --eval-set data/eval_cases/sample_eval.jsonl \
  --strategy beam_search \
  --rounds 3 \
  --beam-width 3 \
  --variants-per-candidate 2 \
  --output-dir data/runs/
```

---

## Storage Approach

- **SQLite** (`data/prompt_evolve.db`) for candidates, eval cases, run results, scores, rounds.
- **JSONL** files under `data/runs/<run_id>/` for raw run artifacts (portable, inspectable).
- **Markdown** files under `data/reports/` for human-readable reports.

---

## Testing Strategy

- `test_scoring.py`: All deterministic scorers with edge cases. No LLM calls.
- `test_selection.py`: Hill climbing and beam search with synthetic score tables.
- `test_mutations.py`: Parameter mutation (`vary_model_parameters`) without LLM; LLM-based mutations with a mocked gateway.
- `test_runner.py`: EvalRunner with a mocked gateway; verify JSONL artifacts written.
- `conftest.py`: Shared fixtures (temp DB, sample candidates, sample eval cases).

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LLM latency in tests | Mock gateway for all unit/integration tests |
| API key absent in CI | `--dry-run` flag; tests mock gateway |
| SQLite concurrency | Single-process CLI; no concurrency needed for MVP |
| Mutation producing invalid output | Validate and retry up to 3 times; raise if still invalid |
| Token cost blowup | `max_tokens` cap; cost tracking surfaced in report |
| Non-determinism | Fixed random seed option; fixed eval set |

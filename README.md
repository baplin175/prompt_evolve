# prompt_evolve

**Prompt Evolution Rig** — an eval-driven system for evolving prompts toward the best-performing variant using automated evaluation, mutation, scoring, and selection.

---

## Overview

Prompt Evolve helps you systematically improve LLM prompts by:

1. Starting from a **baseline prompt**
2. Generating **variants** via controlled mutation operators
3. **Evaluating** all variants on a fixed eval set
4. **Scoring** outputs with deterministic and/or LLM-judge scorers
5. **Selecting** the best candidates (hill climbing or beam search)
6. **Repeating** for configurable rounds
7. **Reporting** the best prompt, leaderboard, and failure analysis

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Configure

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Initialize

```bash
prompt-evolve init
```

### 4. Add a baseline prompt

```bash
prompt-evolve prompt add --file data/prompts/baseline.json
```

### 5. List candidates

```bash
prompt-evolve prompt list
```

### 6. Dry-run evaluation (no API calls)

```bash
prompt-evolve run --eval-set data/eval_cases/sample_eval.jsonl --dry-run
```

### 7. Real evaluation (requires API key)

```bash
prompt-evolve run --eval-set data/eval_cases/sample_eval.jsonl
```

### 8. Score a run

```bash
prompt-evolve score --run-id <run-id> --config data/scoring_config.json
```

### 9. Generate a report

```bash
prompt-evolve report --run-id <run-id> --output data/reports/my_report.md
```

### 10. Run the full optimization loop

```bash
prompt-evolve optimize \
  --baseline data/prompts/baseline.json \
  --eval-set data/eval_cases/sample_eval.jsonl \
  --strategy beam_search \
  --rounds 3 \
  --beam-width 3 \
  --variants-per-candidate 2 \
  --operators simplify_wording,tighten_constraints,add_examples
```

---

## Mutation Operators

| Operator | Description |
|---|---|
| `simplify_wording` | Simplifies language and removes jargon |
| `tighten_constraints` | Adds more explicit constraints |
| `improve_formatting_instructions` | Improves output format instructions |
| `add_examples` | Adds 1–2 few-shot examples |
| `handle_ambiguity` | Clarifies ambiguous or underspecified parts |
| `reduce_verbosity` | Removes redundancy |
| `reorder_instruction_blocks` | Reorders for better logical flow |
| `vary_model_parameters` | Adjusts temperature (no LLM call needed) |

### Mutate a specific candidate

```bash
prompt-evolve mutate \
  --candidate-id <uuid> \
  --operators simplify_wording,add_examples \
  --n 3
```

---

## Scoring Dimensions

| Dimension | Default Weight | Description |
|---|---|---|
| `correctness` | 0.50 | Exact/substring/regex match, judge scoring |
| `robustness` | 0.20 | Error rate across eval cases |
| `format_compliance` | 0.20 | JSON field presence, length constraints |
| `latency_efficiency` | 0.05 | Inverse-normalized latency |
| `cost_efficiency` | 0.05 | Inverse-normalized token cost |

Weights are configurable in `data/scoring_config.json`.

---

## Scorers

### Deterministic Scorers

| Scorer | Config Keys |
|---|---|
| `exact_match` | `case_sensitive` (default: true) |
| `substring_match` | `case_sensitive` (default: true) |
| `regex_match` | `pattern`, `ignore_case` |
| `json_field_presence` | `required_fields: [...]` |
| `length_constraint` | `min_chars`, `max_chars` |
| `custom_python` | `module_path`, `function_name` |

### Model Judge Scorer

Add to `scorers` in your scoring config:

```json
{
  "name": "model_judge",
  "dimension": "correctness",
  "weight": 1.0,
  "task_description": "Answer factual questions accurately.",
  "criteria": {
    "accuracy": {"weight": 0.6, "description": "Is the answer correct?"},
    "clarity": {"weight": 0.4, "description": "Is the answer clearly expressed?"}
  }
}
```

---

## Selection Strategies

| Strategy | Description |
|---|---|
| `hill_climbing` | Keep the single best candidate per round |
| `beam_search` | Keep top-K candidates per round (configurable `--beam-width`) |

---

## Project Structure

```
app/
  cli.py                    # Click CLI entrypoints
  config.py                 # Environment configuration
  models/                   # Pydantic data models
  gateway/                  # LLM gateway abstraction
  mutations/                # Mutation operators and engine
  runner/                   # Eval runner
  scoring/                  # Scorers and aggregator
  optimization/             # Hill climbing, beam search, evolution loop
  reporting/                # Leaderboard and markdown report generator
  storage/                  # SQLite schema and JSONL artifacts
data/
  eval_cases/sample_eval.jsonl    # 12 diverse eval cases
  prompts/baseline.json           # Sample baseline prompt
  scoring_config.json             # Default scoring config
  runs/                           # Runtime: JSONL run artifacts
  reports/                        # Runtime: generated markdown reports
tests/
  conftest.py
  test_scoring.py
  test_selection.py
  test_mutations.py
  test_runner.py
```

---

## Configuration

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for real LLM calls |
| `OPENAI_BASE_URL` | OpenAI | Custom base URL (Azure, Ollama, etc.) |
| `DEFAULT_MODEL` | `gpt-4o-mini` | Default model |
| `DEFAULT_MAX_TOKENS` | `1024` | Default max tokens |
| `LLM_TIMEOUT_SECONDS` | `60` | Request timeout |
| `LLM_MAX_RETRIES` | `3` | Retries on transient errors |
| `DB_PATH` | `data/prompt_evolve.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Log level |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run without an API key (gateway calls are mocked).

---

## Known Limitations

- Optimization loop does not yet support parallel evaluation (sequential only).
- The `custom_python` scorer loads modules from disk — use with trusted code only.
- No built-in retry deduplication across runs (re-run may re-evaluate the same candidates).
- Cost tracking requires the LLM provider to return usage data.

---

## Next Improvements

1. Add async/parallel evaluation for faster runs.
2. Add a web UI for interactive exploration.
3. Support multi-turn conversation eval cases.
4. Add A/B significance testing for score comparisons.
5. Export best prompt to a simple JSON artifact for deployment.

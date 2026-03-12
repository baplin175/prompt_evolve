"""JSONL artifact writer/reader for run results and eval cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append records to a JSONL file (creates file + parent dirs if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")
    logger.debug("Wrote %d records to %s", len(records), path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate over records in a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", lineno, path, exc)


def load_eval_cases_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all eval case records from a JSONL file."""
    return list(read_jsonl(path))


def write_run_artifacts(run_dir: Path, results: list[dict[str, Any]]) -> Path:
    """Write run results to <run_dir>/results.jsonl."""
    artifact_path = run_dir / "results.jsonl"
    write_jsonl(artifact_path, results)
    return artifact_path

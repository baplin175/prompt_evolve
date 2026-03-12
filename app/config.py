"""Configuration loaded from environment variables / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


OPENAI_API_KEY: str = _get("OPENAI_API_KEY")
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL") or None

DEFAULT_MODEL: str = _get("DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS: int = int(_get("DEFAULT_MAX_TOKENS", "1024"))
LLM_TIMEOUT_SECONDS: float = float(_get("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES: int = int(_get("LLM_MAX_RETRIES", "3"))

DB_PATH: Path = Path(_get("DB_PATH", "data/prompt_evolve.db"))
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")

"""Matcha API gateway client with retries and timeout."""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from app import config
from app.gateway.base import GatewayClient, GatewayResponse

logger = logging.getLogger(__name__)


def _extract_reply_text(data: object) -> str:
    """Extract the text content from a Matcha API response."""
    if not isinstance(data, dict):
        return ""

    output = data.get("output", "")
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            content = first.get("content", [])
            if content and isinstance(content[0], dict):
                return str(content[0].get("text", ""))

    if isinstance(output, list):
        return "\n".join(str(item) for item in output)

    return str(output) if output is not None else ""


class MatchaClient(GatewayClient):
    """Gateway client for the Matcha completions API.

    Sends prompts to a Matcha endpoint and returns structured responses
    compatible with the rest of the prompt-evolve pipeline.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        mission_id: Optional[str] = None,
        api_key_header: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[int] = None,
    ) -> None:
        self._url = url or config.MATCHA_URL
        self._api_key = api_key or config.MATCHA_API_KEY
        self._mission_id = mission_id or config.MATCHA_MISSION_ID
        self._api_key_header = api_key_header or config.MATCHA_API_KEY_HEADER
        self._timeout = timeout or config.LLM_TIMEOUT_SECONDS
        self._max_retries = max_retries if max_retries is not None else config.LLM_MAX_RETRIES
        self._retry_backoff = retry_backoff if retry_backoff is not None else 10

    def complete(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        user_content: str,
    ) -> GatewayResponse:
        """Send a completion request to the Matcha API with retry/backoff."""
        prompt = user_content
        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_content}"

        headers = {
            "Content-Type": "application/json",
            self._api_key_header: self._api_key,
        }
        payload = {
            "mission_id": self._mission_id,
            "input": prompt,
        }

        last_error: Optional[Exception] = None
        t0 = time.perf_counter()

        for attempt in range(1, self._max_retries + 1):
            try:
                response = requests.post(
                    self._url,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                latency_ms = (time.perf_counter() - t0) * 1000.0

                data = response.json()
                content = _extract_reply_text(data)

                logger.debug(
                    "Matcha response: %d chars, %.0fms",
                    len(content),
                    latency_ms,
                )

                return GatewayResponse(
                    content=content,
                    model=model,
                    latency_ms=latency_ms,
                    raw=data if isinstance(data, dict) else {},
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = self._retry_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Matcha call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt,
                        self._max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)

        raise last_error  # type: ignore[misc]

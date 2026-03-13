"""OpenAI-compatible gateway client with retries and timeout."""

from __future__ import annotations

import logging
import time
from typing import Optional

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from typing import Any

from app import config
from app.gateway.base import GatewayClient, GatewayResponse

logger = logging.getLogger(__name__)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient(GatewayClient):
    """OpenAI-compatible gateway client.

    Works with any provider that implements the OpenAI chat completions API
    (OpenAI, Azure OpenAI, Ollama, LM Studio, Together AI, etc.).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or config.OPENAI_API_KEY
        self._base_url = base_url or config.OPENAI_BASE_URL
        self._timeout = timeout or config.LLM_TIMEOUT_SECONDS
        self._max_retries = max_retries or config.LLM_MAX_RETRIES
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,  # We handle retries via tenacity
        )

    def complete(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        user_content: str,
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> GatewayResponse:
        """Send a chat completion request with retry/backoff."""
        return self._complete_with_retry(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_content=user_content,
            messages=messages,
        )

    def _complete_with_retry(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        user_content: str,
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> GatewayResponse:
        """Inner method so we can apply tenacity retry decorator dynamically."""
        attempt_fn = retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )(self._do_complete)

        return attempt_fn(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_content=user_content,
            messages=messages,
        )

    def _do_complete(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        user_content: str,
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> GatewayResponse:
        if messages is not None:
            chat_messages = list(messages)
        else:
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": user_content})

        logger.debug("Calling %s (temp=%.2f, max_tokens=%d)", model, temperature, max_tokens)

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        logger.debug(
            "Response from %s: %d chars, %.0fms, in=%s out=%s tokens",
            model,
            len(content),
            latency_ms,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
        )

        return GatewayResponse(
            content=content,
            model=response.model,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

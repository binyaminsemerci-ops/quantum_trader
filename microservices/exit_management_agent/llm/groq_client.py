"""PATCH-11 — Groq HTTP client with per-model rate throttle.

Single responsibility: POST /v1/chat/completions to Groq and return raw str.
Each GroqModelClient instance owns its own rate-throttle state.
Instantiate one per model (primary, fallback, evaluator).

Raises on HTTP errors / timeouts — callers handle fallback logic.
Never logs the api_key.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

_log = logging.getLogger("exit_management_agent.llm.groq_client")

_GROQ_CHAT_PATH = "/chat/completions"


class ThrottleSkipError(Exception):
    """Raised when the per-model rate throttle fires."""


class GroqModelClient:
    """
    Async HTTP client for one Groq model.

    Args:
        endpoint:          Groq base URL, e.g. "https://api.groq.com/openai/v1"
        model:             Groq model ID, e.g. "qwen/qwen3-32b"
        api_key:           Groq API key — never logged
        timeout_ms:        Per-request timeout (clamped [200, 30000])
        min_interval_sec:  Rate throttle — min seconds between calls (0 = off)
        temperature:       Sampling temperature (default 0.2 for determinism)
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_ms: int = 5000,
        min_interval_sec: float = 3.0,
        temperature: float = 0.2,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._chat_url = self._endpoint + _GROQ_CHAT_PATH
        self._model = model
        self._api_key = api_key  # never logged
        self._timeout_sec = max(0.2, min(30.0, timeout_ms / 1000.0))
        self._min_interval_sec = max(0.0, min_interval_sec)
        self._temperature = temperature
        self._last_call_ts: float = 0.0
        self._call_lock: asyncio.Lock = asyncio.Lock()
        _log.info(
            "[GroqClient] init model=%s timeout=%.1fs throttle=%.1fs",
            self._model,
            self._timeout_sec,
            self._min_interval_sec,
        )

    @property
    def model(self) -> str:
        return self._model

    async def chat(self, system_prompt: str, user_content: str) -> str:
        """
        POST chat/completions and return assistant content string.

        Raises:
            aiohttp.ClientResponseError — HTTP 4xx/5xx (incl. 429)
            asyncio.TimeoutError        — request timed out
            ThrottleSkipError           — rate throttle fired
        """
        if self._min_interval_sec > 0.0:
            async with self._call_lock:
                now = time.monotonic()
                wait = self._min_interval_sec - (now - self._last_call_ts)
                if wait > 0.0:
                    _log.debug(
                        "[GroqClient] throttle wait %.2fs for model=%s",
                        wait, self._model,
                    )
                    await asyncio.sleep(wait)
                self._last_call_ts = time.monotonic()

        request_body = {
            "model": self._model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._chat_url, json=request_body, headers=headers
            ) as resp:
                resp.raise_for_status()
                body = await resp.json(content_type=None)
        return body["choices"][0]["message"]["content"]

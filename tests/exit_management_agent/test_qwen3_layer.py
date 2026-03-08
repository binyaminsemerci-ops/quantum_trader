"""Tests for Qwen3Layer (PATCH-7B / PATCH-7B-ext).

Coverage:
  - Auth header present when api_key is set (Ollama-style and OpenAI-style)
  - Auth header absent when api_key is empty
  - api_key never appears in logs, error reason strings, or result.raw
  - Fallback behavior unchanged: HTTP error, timeout, invalid JSON,
    disallowed action all produce fallback=True with formula action
  - Valid OpenAI-compat response parsed correctly (fallback=False)
  - Valid Ollama response parsed correctly (fallback=False)

No real network calls — aiohttp.ClientSession is replaced with a fake that
records request headers and returns a configurable response.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from microservices.exit_management_agent.models import ExitScoreState
from microservices.exit_management_agent.qwen3_layer import Qwen3Layer


# ── Helpers ────────────────────────────────────────────────────────────────────


def _score_state(formula_action: str = "HOLD") -> ExitScoreState:
    """Minimal ExitScoreState for use in tests."""
    return ExitScoreState(
        symbol="ETHUSDT",
        side="LONG",
        R_net=-0.10,
        age_sec=120.0,
        age_fraction=0.008,
        giveback_pct=0.0,
        distance_to_sl_pct=0.083,
        peak_price=2010.0,
        mark_price=1950.0,
        entry_price=2000.0,
        leverage=1.0,
        r_effective_t1=2.0,
        r_effective_lock=0.0,
        d_r_loss=0.067,
        d_r_gain=0.0,
        d_giveback=0.0,
        d_time=0.0,
        d_sl_proximity=0.0,
        exit_score=0.020,
        formula_action=formula_action,
        formula_urgency="LOW",
        formula_confidence=0.020,
        formula_reason="Score=0.020 — no exit criteria met",
    )


def _layer(endpoint: str = "http://localhost:11434", api_key: str = "") -> Qwen3Layer:
    return Qwen3Layer(
        endpoint=endpoint,
        timeout_ms=3000,
        shadow=True,
        model="qwen3:0.6b",
        api_key=api_key,
    )


def _openai_response_body(action: str = "HOLD", confidence: float = 0.85) -> dict:
    """A minimal valid OpenAI-compat chat completion response."""
    content = json.dumps(
        {"action": action, "confidence": confidence, "reason": "Test reason"}
    )
    return {
        "choices": [{"message": {"content": content}}],
    }


def _ollama_response_body(action: str = "HOLD", confidence: float = 0.75) -> dict:
    """A minimal valid Ollama /api/chat response."""
    content = json.dumps(
        {"action": action, "confidence": confidence, "reason": "Ollama reason"}
    )
    return {"message": {"content": content}}


class _FakeResponse:
    """Fake aiohttp response context manager."""

    def __init__(self, body: dict | None = None, raise_on_raise: Exception | None = None):
        self._body = body or {}
        self._raise = raise_on_raise
        # Records the headers the request was sent with (injected by _FakeSession)
        self.request_headers: dict = {}

    async def json(self, content_type=None):
        return self._body

    def raise_for_status(self):
        if self._raise:
            raise self._raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """
    Fake aiohttp.ClientSession that captures the last request's headers
    and returns a configurable _FakeResponse.
    """

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.captured_headers: dict = {}
        self.captured_url: str = ""

    def post(self, url: str, *, json: Any = None, headers: dict | None = None):
        self.captured_headers = dict(headers or {})
        self.captured_url = url
        self._response.request_headers = self.captured_headers
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _patch_session(fake_session: _FakeSession):
    """Return a context manager that replaces aiohttp.ClientSession."""
    return patch(
        "aiohttp.ClientSession",
        return_value=fake_session,
    )


# ── Auth header tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_header_sent_when_api_key_set_openai_style():
    """Authorization: Bearer <key> is present when api_key is non-empty (OpenAI-compat endpoint)."""
    response = _FakeResponse(_openai_response_body())
    session = _FakeSession(response)
    layer = _layer(endpoint="https://api.groq.com/openai/v1", api_key="sk-secret-key")

    with _patch_session(session):
        result = await layer.evaluate(_score_state())

    assert "Authorization" in session.captured_headers
    assert session.captured_headers["Authorization"] == "Bearer sk-secret-key"
    assert result.fallback is False


@pytest.mark.asyncio
async def test_auth_header_sent_when_api_key_set_ollama_style():
    """Authorization header is also sent for Ollama-style endpoints when api_key is set."""
    response = _FakeResponse(_ollama_response_body())
    session = _FakeSession(response)
    layer = _layer(endpoint="http://localhost:11434", api_key="local-token")

    with _patch_session(session):
        result = await layer.evaluate(_score_state())

    assert session.captured_headers.get("Authorization") == "Bearer local-token"
    assert result.fallback is False


@pytest.mark.asyncio
async def test_no_auth_header_when_api_key_empty():
    """No Authorization header is added when api_key is empty string (Ollama default)."""
    response = _FakeResponse(_ollama_response_body())
    session = _FakeSession(response)
    layer = _layer(endpoint="http://localhost:11434", api_key="")

    with _patch_session(session):
        await layer.evaluate(_score_state())

    assert "Authorization" not in session.captured_headers


@pytest.mark.asyncio
async def test_no_auth_header_when_api_key_not_provided():
    """No Authorization header when Qwen3Layer is constructed without api_key kwarg."""
    response = _FakeResponse(_ollama_response_body())
    session = _FakeSession(response)
    # Construct without api_key — uses default ""
    layer = Qwen3Layer(
        endpoint="http://localhost:11434",
        timeout_ms=3000,
        shadow=True,
        model="qwen3:0.6b",
    )

    with _patch_session(session):
        await layer.evaluate(_score_state())

    assert "Authorization" not in session.captured_headers


# ── Secret safety tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_key_never_appears_in_result_raw_on_success(caplog):
    """api_key must not appear in result.raw or any log output."""
    SECRET = "sk-ultra-secret-9999"
    response = _FakeResponse(_openai_response_body())
    session = _FakeSession(response)
    layer = _layer(endpoint="https://api.groq.com/openai/v1", api_key=SECRET)

    with caplog.at_level(logging.DEBUG, logger="exit_management_agent.qwen3_layer"):
        with _patch_session(session):
            result = await layer.evaluate(_score_state())

    assert SECRET not in result.raw
    assert SECRET not in result.reason
    for record in caplog.records:
        assert SECRET not in record.getMessage()


@pytest.mark.asyncio
async def test_api_key_never_appears_in_logs_on_http_error(caplog):
    """api_key must not appear in logs when the HTTP call raises."""
    import aiohttp
    SECRET = "sk-leaked-if-bug"

    async def _bad_post(*args, **kwargs):
        raise aiohttp.ClientError("connection refused")

    layer = _layer(endpoint="https://api.groq.com/openai/v1", api_key=SECRET)

    # Simulate a network error at the session.post level
    fake_response = _FakeResponse(raise_on_raise=aiohttp.ClientError("refused"))
    session = _FakeSession(fake_response)

    with caplog.at_level(logging.WARNING, logger="exit_management_agent.qwen3_layer"):
        with _patch_session(session):
            result = await layer.evaluate(_score_state())

    assert result.fallback is True
    for record in caplog.records:
        assert SECRET not in record.getMessage()


@pytest.mark.asyncio
async def test_api_key_never_in_error_reason():
    """The error reason string on fallback must not contain the api_key."""
    import aiohttp
    SECRET = "sk-must-not-leak"

    class _ErrorSession:
        def post(self, url, *, json=None, headers=None):
            return _ErrorResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class _ErrorResponse:
        def raise_for_status(self):
            raise aiohttp.ClientError("bad gateway")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    layer = _layer(endpoint="https://api.groq.com/openai/v1", api_key=SECRET)
    with patch("aiohttp.ClientSession", return_value=_ErrorSession()):
        result = await layer.evaluate(_score_state())

    assert SECRET not in result.reason
    assert SECRET not in result.raw


# ── Fallback behavior tests (regressions) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_on_http_timeout():
    """Timeout exception → fallback=True, action=formula_action."""
    import asyncio

    class _TimeoutResponse:
        """Simulates aiohttp timing out when the response context manager is entered."""

        async def __aenter__(self):
            raise asyncio.TimeoutError()

        async def __aexit__(self, *args):
            pass

    class _TimeoutSession:
        def post(self, url, *, json=None, headers=None):
            return _TimeoutResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    layer = _layer(api_key="sk-any")
    with patch("aiohttp.ClientSession", return_value=_TimeoutSession()):
        result = await layer.evaluate(_score_state("HOLD"))

    assert result.fallback is True
    assert result.action == "HOLD"
    assert "TimeoutError" in result.reason
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_fallback_on_invalid_json_response():
    """Non-JSON model response → fallback=True, action=formula_action."""

    class _BadJsonResponse:
        def raise_for_status(self):
            pass

        async def json(self, content_type=None):
            return {"message": {"content": "not json at all %%"}}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class _BadJsonSession:
        def post(self, url, *, json=None, headers=None):
            return _BadJsonResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    layer = _layer()
    with patch("aiohttp.ClientSession", return_value=_BadJsonSession()):
        result = await layer.evaluate(_score_state("PARTIAL_CLOSE_25"))

    assert result.fallback is True
    assert result.action == "PARTIAL_CLOSE_25"
    assert result.reason == "qwen3_parse_error"


@pytest.mark.asyncio
async def test_fallback_on_disallowed_action():
    """Model returns an action not in _ALLOWED_ACTIONS → fallback=True."""
    bad_body = {"message": {"content": json.dumps(
        {"action": "TIGHTEN_TRAIL", "confidence": 0.9, "reason": "Trail it"}
    )}}

    class _DisallowedSession:
        def post(self, url, *, json=None, headers=None):
            return _FakeResponse(bad_body)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    layer = _layer()
    with patch("aiohttp.ClientSession", return_value=_DisallowedSession()):
        result = await layer.evaluate(_score_state("FULL_CLOSE"))

    assert result.fallback is True
    assert result.action == "FULL_CLOSE"
    assert "TIGHTEN_TRAIL" in result.reason


@pytest.mark.asyncio
async def test_valid_openai_response_no_fallback():
    """Valid OpenAI-compat response → fallback=False, correct fields parsed."""
    body = _openai_response_body(action="FULL_CLOSE", confidence=0.92)
    response = _FakeResponse(body)
    session = _FakeSession(response)
    layer = _layer(endpoint="https://api.groq.com/openai/v1", api_key="sk-x")

    with _patch_session(session):
        result = await layer.evaluate(_score_state())

    assert result.fallback is False
    assert result.action == "FULL_CLOSE"
    assert abs(result.confidence - 0.92) < 1e-6
    assert result.reason == "Test reason"
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_valid_ollama_response_no_fallback():
    """Valid Ollama response → fallback=False, correct fields parsed."""
    body = _ollama_response_body(action="PARTIAL_CLOSE_25", confidence=0.60)
    response = _FakeResponse(body)
    session = _FakeSession(response)
    layer = _layer(endpoint="http://localhost:11434", api_key="")

    with _patch_session(session):
        result = await layer.evaluate(_score_state())

    assert result.fallback is False
    assert result.action == "PARTIAL_CLOSE_25"
    assert abs(result.confidence - 0.60) < 1e-6


# ── URL routing tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_compat_url_used_for_v1_endpoint():
    """/v1 in endpoint → POST goes to /v1/chat/completions."""
    body = _openai_response_body()
    response = _FakeResponse(body)
    session = _FakeSession(response)
    layer = _layer(endpoint="https://api.groq.com/openai/v1", api_key="sk-k")

    with _patch_session(session):
        await layer.evaluate(_score_state())

    assert session.captured_url.endswith("/chat/completions")


@pytest.mark.asyncio
async def test_ollama_url_used_for_local_endpoint():
    """Local Ollama endpoint → POST goes to /api/chat."""
    body = _ollama_response_body()
    response = _FakeResponse(body)
    session = _FakeSession(response)
    layer = _layer(endpoint="http://localhost:11434", api_key="")

    with _patch_session(session):
        await layer.evaluate(_score_state())

    assert session.captured_url.endswith("/api/chat")

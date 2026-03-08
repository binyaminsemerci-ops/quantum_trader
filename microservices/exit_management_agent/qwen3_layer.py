"""qwen3_layer: PATCH-7B — Qwen3 constrained decision layer.

Architecture
------------
Qwen3 sits AFTER the formula engine and BEFORE the ExitDecision is built.
The call path is:

    HardGuards.evaluate()     ← fires first; bypasses everything if triggered
    ScoringEngine.score()     ← always runs when no hard guard fired
    Qwen3Layer.evaluate()     ← called only in scoring_mode="ai"
    ExitDecision(...)         ← built from qwen3 result (or formula fallback)

Constraints
-----------
- Qwen3 may only select from _ALLOWED_ACTIONS (4 values).
  TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN are *not* available; those actions require
  exact SL price computation done outside the scoring engine.  main.py skips
  Qwen3 entirely for those formula actions.
- On ANY failure — HTTP error, timeout, invalid JSON, disallowed action — the
  fallback is the formula engine's recommendation, not the legacy rule chain.
- qwen3_shadow=True (default): model output is written to audit but the LIVE
  decision path uses the formula recommendation unchanged.  No gateway or
  ownership-flag changes.
- Never raises — all exceptions are caught and produce a fallback result.

Input: structured JSON derived from ExitScoreState (14 numeric fields + 1 dict).
Output: Qwen3LayerResult (action, confidence, reason, fallback, latency_ms, raw).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import aiohttp

_log = logging.getLogger("exit_management_agent.qwen3_layer")

# Actions Qwen3 is allowed to select.
# TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN are excluded — they require exact SL
# price arithmetic computed from PerceptionResult data not sent to the model.
_ALLOWED_ACTIONS: frozenset = frozenset(
    {"HOLD", "PARTIAL_CLOSE_25", "FULL_CLOSE", "TIME_STOP_EXIT"}
)

_MAX_REASON_LEN: int = 200
_RAW_CAPTURE_LEN: int = 500

_SYSTEM_PROMPT: str = (
    "You are a constrained trading risk advisor operating on Binance testnet futures.\n"
    "You receive a JSON object describing one open position and the formula engine's\n"
    "exit recommendation. Your only job is to select an exit action.\n\n"
    "Allowed actions (you MUST use exactly one of these strings):\n"
    "  HOLD\n"
    "  PARTIAL_CLOSE_25\n"
    "  FULL_CLOSE\n"
    "  TIME_STOP_EXIT\n\n"
    "Rules:\n"
    "- Emergency stops are already handled upstream. You will never see urgency=EMERGENCY.\n"
    "- TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN are not available to you.\n"
    "- If uncertain, prefer HOLD (defer to the formula recommendation).\n"
    "- formula_suggestion is advisory. You may agree or override it.\n\n"
    "You MUST respond with ONLY a JSON object — no markdown, no explanation, no extra text:\n"
    '{"action": "<one of the 4 actions>", "confidence": <0.0-1.0>, "reason": "<max 120 chars>"}'
)


def _build_payload(ss: "ExitScoreState") -> dict:  # type: ignore[name-defined]
    """Construct the structured JSON input for the model from ExitScoreState."""
    return {
        "symbol": ss.symbol,
        "side": ss.side,
        "R_net": round(ss.R_net, 4),
        "age_sec": round(ss.age_sec, 1),
        "age_fraction": round(ss.age_fraction, 4),
        "giveback_pct": round(ss.giveback_pct, 4),
        "distance_to_sl_pct": round(ss.distance_to_sl_pct, 4),
        "leverage": round(ss.leverage, 1),
        "exit_score": round(ss.exit_score, 4),
        "d_r_loss": round(ss.d_r_loss, 4),
        "d_r_gain": round(ss.d_r_gain, 4),
        "d_giveback": round(ss.d_giveback, 4),
        "d_time": round(ss.d_time, 4),
        "d_sl_proximity": round(ss.d_sl_proximity, 4),
        "formula_suggestion": {
            "action": ss.formula_action,
            "urgency": ss.formula_urgency,
            "confidence": round(ss.formula_confidence, 4),
            "reason": ss.formula_reason,
        },
    }


def _parse_response(raw: str, fallback_action: str) -> "Qwen3LayerResult":  # type: ignore[name-defined]
    """
    Parse a raw model response string into Qwen3LayerResult.

    On any parse / validation error returns a fallback result with
    fallback=True and action=fallback_action.
    """
    from .models import Qwen3LayerResult

    raw_captured = raw[:_RAW_CAPTURE_LEN]

    try:
        parsed = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        _log.warning("[Qwen3] JSON parse error: %s  raw=%r", exc, raw_captured)
        return Qwen3LayerResult(
            action=fallback_action,
            confidence=0.0,
            reason="qwen3_parse_error",
            fallback=True,
            latency_ms=0.0,
            raw=raw_captured,
        )

    action = parsed.get("action", "")
    if action not in _ALLOWED_ACTIONS:
        _log.warning(
            "[Qwen3] Disallowed action %r — falling back to formula=%r",
            action,
            fallback_action,
        )
        return Qwen3LayerResult(
            action=fallback_action,
            confidence=0.0,
            reason=f"qwen3_disallowed_action:{action}",
            fallback=True,
            latency_ms=0.0,
            raw=raw_captured,
        )

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", ""))[:_MAX_REASON_LEN]

    return Qwen3LayerResult(
        action=action,
        confidence=confidence,
        reason=reason,
        fallback=False,
        latency_ms=0.0,  # caller fills this in
        raw=raw_captured,
    )


class Qwen3Layer:
    """
    Async HTTP client for the Qwen3 inference endpoint.

    Supports both Ollama (/api/chat) and OpenAI-compatible (/v1/chat/completions)
    endpoints.  The endpoint type is inferred from the path: if the base URL's
    path is empty or '/', Ollama-style is used; otherwise the full URL is used
    as-is (override by passing an explicit full path).

    Never raises — all I/O errors are caught and produce fallback=True results.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_ms: int,
        shadow: bool,
        model: str = "qwen3:8b",
        api_key: str = "",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout_sec = max(0.2, timeout_ms / 1000.0)
        self._shadow = shadow
        self._model = model
        # PATCH-7B-ext: bearer token; empty = no header (local Ollama default).
        # Never passed to _log or included in any result/error string.
        self._api_key = api_key

        # Determine chat URL from endpoint.
        # Ollama default: POST /api/chat
        # OpenAI-compat: POST /v1/chat/completions
        if "/v1" in self._endpoint:
            self._chat_url = self._endpoint + "/chat/completions"
            self._api_style = "openai"
        else:
            self._chat_url = self._endpoint + "/api/chat"
            self._api_style = "ollama"

        _log.info(
            "[Qwen3] Initialised endpoint=%s model=%s timeout=%.1fs shadow=%s style=%s",
            self._endpoint,
            self._model,
            self._timeout_sec,
            self._shadow,
            self._api_style,
        )

    async def evaluate(self, score_state: "ExitScoreState") -> "Qwen3LayerResult":  # type: ignore[name-defined]
        """
        Call Qwen3 with structured score_state input.

        Returns Qwen3LayerResult.  On any failure (network, parse, validation),
        returns a result with fallback=True and action=score_state.formula_action.

        Never raises.
        """
        from .models import Qwen3LayerResult

        fallback_action = score_state.formula_action
        payload = _build_payload(score_state)
        user_content = json.dumps(payload, separators=(",", ":"))

        t0 = time.monotonic()
        try:
            raw = await self._post(user_content)
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            _log.warning(
                "[Qwen3] HTTP error after %.0fms: %s — using formula fallback=%r",
                latency_ms,
                exc,
                fallback_action,
            )
            return Qwen3LayerResult(
                action=fallback_action,
                confidence=0.0,
                reason=f"qwen3_http_error:{type(exc).__name__}",
                fallback=True,
                latency_ms=latency_ms,
                raw="",
            )

        latency_ms = (time.monotonic() - t0) * 1000.0
        result = _parse_response(raw, fallback_action)
        # Inject real latency (parse helper sets 0.0 as placeholder)
        result = Qwen3LayerResult(
            action=result.action,
            confidence=result.confidence,
            reason=result.reason,
            fallback=result.fallback,
            latency_ms=latency_ms,
            raw=result.raw,
        )

        _log.debug(
            "[Qwen3] symbol=%s action=%s confidence=%.2f fallback=%s latency=%.0fms",
            score_state.symbol,
            result.action,
            result.confidence,
            result.fallback,
            result.latency_ms,
        )
        return result

    async def _post(self, user_content: str) -> str:
        """
        POST to the inference endpoint and return the assistant message content.

        Uses aiohttp.  Raises on HTTP errors or timeouts so the caller can
        catch and build a fallback result.
        """
        request_body: dict
        if self._api_style == "ollama":
            request_body = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "format": "json",
            }
        else:
            request_body = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
            }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._chat_url,
                json=request_body,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                body = await resp.json(content_type=None)

        # Extract assistant content string from either API style
        if self._api_style == "ollama":
            content = body["message"]["content"]
        else:
            content = body["choices"][0]["message"]["content"]

        return content

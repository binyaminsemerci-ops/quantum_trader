"""ai_brain: PATCH-11 — 3-tier AI decision and evaluation layer.

Live path  (called per tick):
    AIBrain.evaluate(score_state) -> Qwen3LayerResult
    Tier 1 — PRIMARY:   Qwen3-32b           (Groq) — live exit decisions
    Tier 2 — FALLBACK:  Mistral Small 3.1   (Groq) — takes over on Tier 1 failure / 429
    Tier 0 — FORMULA:   formula engine      in main — last resort (qr.fallback=True)

Offline path (called from ReplayWriter after position close):
    DeepSeekEvaluator.evaluate_replay(record) -> dict
    Tier 3 — EVALUATOR: DeepSeek-R1-Distill-70b (Groq) — "was the decision correct?"
    Returns deepseek_* fields merged into the replay record.
    Never called on the live tick path — offline only.
"""
from __future__ import annotations

import json
import logging
import re
import time

import aiohttp

from .qwen3_layer import Qwen3Layer

_log = logging.getLogger("exit_management_agent.ai_brain")

_EVAL_SYSTEM_PROMPT = (
    "You are a post-trade decision auditor. You receive a JSON record of a completed "
    "futures trade including: the action taken (live_action), formula suggestion "
    "(formula_action), outcome metrics (hold_duration_sec, closed_by, reward, "
    "regret_label, preferred_action).\n\n"
    "Your job: was live_action the best possible decision at that moment?\n\n"
    "Allowed actions: HOLD, PARTIAL_CLOSE_25, PARTIAL_CLOSE_50, FULL_CLOSE, "
    "TIME_STOP_EXIT\n\n"
    "Respond with ONLY a JSON object — no markdown, no extra text:\n"
    '{"was_correct": true/false, "better_action": "<action>", '
    '"confidence": <0.0-1.0>, "reasoning": "<max 150 chars>"}'
)

_MAX_REASON_LEN: int = 200


def _strip_think(text: str) -> str:
    """Strip DeepSeek-R1 <think>...</think> reasoning blocks before JSON parse."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class AIBrain:
    """
    3-tier live AI decision layer.

    Tier 1 (primary):  Qwen3-32b    via Groq — drives live exit decisions
    Tier 2 (fallback): Mistral 3.1  via Groq — activates on Tier 1 failure / 429
    Tier 0 (formula):  formula eng. in main  — last resort (qr.fallback=True)

    Returns Qwen3LayerResult for drop-in compatibility with existing main.py.
    The reason field is prefixed with 't1:', 't2:', or 't0:' to identify which tier fired.
    """

    def __init__(self, primary: Qwen3Layer, fallback: Qwen3Layer) -> None:
        self._primary = primary
        self._fallback = fallback

    async def evaluate(self, score_state) -> "Qwen3LayerResult":  # type: ignore[name-defined]
        from .models import Qwen3LayerResult

        # ── Tier 1: Qwen3-32b ────────────────────────────────────────────────
        result = await self._primary.evaluate(score_state)
        if not result.fallback:
            return Qwen3LayerResult(
                action=result.action,
                confidence=result.confidence,
                reason=f"t1:{result.reason}",
                fallback=False,
                latency_ms=result.latency_ms,
                raw=result.raw,
            )

        _log.warning(
            "[AIBrain] Tier-1 (Qwen3) fallback reason=%r — trying Tier-2 (Mistral)",
            result.reason,
        )

        # ── Tier 2: Mistral Small 3.1 ────────────────────────────────────────
        result2 = await self._fallback.evaluate(score_state)
        if not result2.fallback:
            return Qwen3LayerResult(
                action=result2.action,
                confidence=result2.confidence,
                reason=f"t2:{result2.reason}",
                fallback=False,
                latency_ms=result2.latency_ms,
                raw=result2.raw,
            )

        _log.warning(
            "[AIBrain] Tier-2 (Mistral) also fallback reason=%r — formula takes over",
            result2.reason,
        )

        # ── Tier 0: both AI tiers failed — formula takes over in main.py ─────
        return Qwen3LayerResult(
            action=result2.action,   # = formula_action set by Qwen3Layer on fallback
            confidence=0.0,
            reason="t0:both_ai_tiers_failed",
            fallback=True,
            latency_ms=0.0,
            raw="",
        )


class DeepSeekEvaluator:
    """
    Offline Tier-3 evaluator — called from ReplayWriter after a position closes.

    Uses DeepSeek-R1-Distill-llama-70b via Groq.
    Answers: "was this exit decision correct?" by analysing the full trade outcome.
    Never raises — all exceptions are caught and produce a fallback dict.
    Never called on the live tick path — offline only.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_ms: int,
        enabled: bool = True,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._chat_url = self._endpoint + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout_sec = max(1.0, timeout_ms / 1000.0)
        self._enabled = enabled
        _log.info(
            "[DeepSeekEval] Initialised model=%s timeout=%.1fs enabled=%s",
            self._model,
            self._timeout_sec,
            self._enabled,
        )

    async def evaluate_replay(self, record: dict) -> dict:
        """
        Post-trade DeepSeek evaluation of a completed replay record.

        Args:
            record: replay record dict (live_action, formula_action, reward,
                    regret_label, preferred_action, hold_duration_sec, etc.)

        Returns:
            dict with deepseek_* keys to be merged into the replay record.
            On any failure returns deepseek_fallback='true' dict — never raises.
        """
        if not self._enabled:
            return {"deepseek_fallback": "true", "deepseek_skip_reason": "disabled"}

        payload = {
            "live_action": record.get("live_action", "UNKNOWN"),
            "formula_action": record.get("formula_action", "null"),
            "reward": record.get("reward", "null"),
            "regret_label": record.get("regret_label", "null"),
            "preferred_action": record.get("preferred_action", "null"),
            "hold_duration_sec": record.get("hold_duration_sec", "null"),
            "closed_by": record.get("closed_by", "unknown"),
            "exit_score": record.get("exit_score", "null"),
            "symbol": record.get("symbol", "UNKNOWN"),
            "side": record.get("side", "UNKNOWN"),
        }

        t0 = time.monotonic()
        try:
            raw = await self._call_api(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            _log.warning("[DeepSeekEval] HTTP error after %.0fms: %s", latency_ms, exc)
            return {
                "deepseek_fallback": "true",
                "deepseek_latency_ms": f"{latency_ms:.0f}",
                "deepseek_error": type(exc).__name__[:40],
            }

        latency_ms = (time.monotonic() - t0) * 1000.0
        result = self._parse(raw, latency_ms)
        _log.info(
            "[DeepSeekEval] symbol=%s was_correct=%s better_action=%s "
            "confidence=%s latency=%.0fms",
            record.get("symbol", "?"),
            result.get("deepseek_was_correct"),
            result.get("deepseek_better_action"),
            result.get("deepseek_confidence"),
            latency_ms,
        )
        return result

    async def _call_api(self, user_content: str) -> str:
        # No response_format — DeepSeek on Groq may not support json_object mode.
        # Prompt instructs JSON output; _parse strips <think> blocks and parses.
        request_body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._chat_url, json=request_body, headers=headers
            ) as resp:
                resp.raise_for_status()
                body = await resp.json(content_type=None)
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(raw: str, latency_ms: float) -> dict:
        """Parse DeepSeek response — strips <think> blocks first."""
        cleaned = _strip_think(raw)
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {
                "deepseek_fallback": "true",
                "deepseek_latency_ms": f"{latency_ms:.0f}",
                "deepseek_error": "parse_error",
                "deepseek_raw": raw[:200],
            }

        was_correct = str(parsed.get("was_correct", "")).lower()
        if was_correct not in ("true", "false"):
            was_correct = "unknown"

        try:
            confidence = f"{float(parsed.get('confidence', 0.5)):.4f}"
        except (TypeError, ValueError):
            confidence = "0.5000"

        return {
            "deepseek_was_correct": was_correct,
            "deepseek_better_action": str(parsed.get("better_action", "UNKNOWN"))[:30],
            "deepseek_confidence": confidence,
            "deepseek_reasoning": str(parsed.get("reasoning", ""))[:_MAX_REASON_LEN],
            "deepseek_latency_ms": f"{latency_ms:.0f}",
            "deepseek_fallback": "false",
        }

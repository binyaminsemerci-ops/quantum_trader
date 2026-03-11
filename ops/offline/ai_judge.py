"""ai_judge: PATCH-11 — 3-layer live model judge + offline evaluator.

Live tick path:
    AIJudge.evaluate(score_state) -> JudgeResult
    Layer 1: Primary  — qwen/qwen3-32b
    Layer 2: Fallback — openai/gpt-oss-20b
    Layer 0: Formula  — deterministic safe degrade on double failure

Offline path (replay/obituary only):
    OfflineEvaluator.evaluate_replay(record) -> dict
    Uses llama-3.3-70b-versatile (never on live tick path)

Failover rules (per PATCH-11 spec):
  A. NORMAL:       Primary succeeds + valid + confidence >= threshold -> proceed
  B. HARD FAILURE: timeout / 429 / 5xx / invalid JSON / illegal action -> fallback
  C. SOFT FAILURE: low confidence / internal conflict / second-opinion policy -> fallback as second opinion
  D. DOUBLE FAIL:  both tiers fail -> deterministic safe degrade, no LLM blocked loop

Shadow rollout phases controlled by AIJudgeConfig.shadow_mode:
  "shadow"  — model runs audit-only; formula is source of truth
  "hybrid"  — model drives soft actions only (HOLD/REDUCE_25/DEFENSIVE_HOLD/HARVEST_70_KEEP_30)
  "live"    — model drives all allowed actions

Compatibility:
  JudgeResult.as_qwen3_layer_result() returns Qwen3LayerResult for backward
  compatibility with existing main.py wiring until main.py is fully migrated.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .groq_client import GroqModelClient, ThrottleSkipError
from .judge_validator import validate, ValidationResult, LIVE_ACTIONS

_log = logging.getLogger("exit_management_agent.ai_judge")

# ── Shadow-mode soft action set (Phase B) ────────────────────────────────────
_HYBRID_ALLOWED: frozenset = frozenset({
    "HOLD",
    "REDUCE_25",
    "DEFENSIVE_HOLD",
    "HARVEST_70_KEEP_30",
})

# ── Confidence threshold to proceed without second opinion ───────────────────
_CONFIDENCE_THRESHOLD: float = 0.55

# ── Conservative disagreement resolver ───────────────────────────────────────
_MIDDLE_ACTION: dict = {
    # (primary, fallback) -> middle policy action
    ("HOLD", "FULL_CLOSE"):          "REDUCE_25",
    ("FULL_CLOSE", "HOLD"):          "REDUCE_25",
    ("HOLD", "TOXICITY_UNWIND"):     "REDUCE_50",
    ("TOXICITY_UNWIND", "HOLD"):     "REDUCE_50",
    ("FULL_CLOSE", "HARVEST_70_KEEP_30"): "HARVEST_70_KEEP_30",
    ("HARVEST_70_KEEP_30", "FULL_CLOSE"): "HARVEST_70_KEEP_30",
}
_DEFAULT_MIDDLE: str = "DEFENSIVE_HOLD"

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a constrained exit-policy judge for a live futures trading system.\n"
    "You receive a structured JSON object containing scoring signals for one open position.\n\n"
    "Your ONLY task: choose ONE exit action from the allowed set below.\n\n"
    "Allowed actions (use EXACTLY one of these strings):\n"
    "  HOLD\n"
    "  REDUCE_25\n"
    "  REDUCE_50\n"
    "  HARVEST_70_KEEP_30\n"
    "  FULL_CLOSE\n"
    "  DEFENSIVE_HOLD\n"
    "  TOXICITY_UNWIND\n"
    "  QUARANTINE\n\n"
    "Allowed reason codes (use 1–5 from this list only):\n"
    "  THESIS_DECAY, TOXICITY_RISING, PAYOUT_FLATTENING, ALPHA_DEPLETED,\n"
    "  CONVICTION_BUDGET_LOW, REGIME_DRIFT, GIVEBACK_EXCESSIVE, SL_PROXIMITY,\n"
    "  TIME_LIMIT_APPROACHING, PROFIT_TARGET_REACHED, LOSS_PRESSURE,\n"
    "  VELOCITY_NEGATIVE, ACCELERATION_NEGATIVE, COUNCIL_CLOSE,\n"
    "  KILL_CHAIN_ELEVATED, NARRATIVE_COLLAPSE, LOW_CONFIDENCE,\n"
    "  FORMULA_OVERRIDE, PORTFOLIO_RISK\n\n"
    "Rules:\n"
    "- Emergency stops are handled upstream. You will never see urgency=EMERGENCY.\n"
    "- FLIP_CANDIDATE is NOT an allowed action.\n"
    "- Do not invent execution parameters (quantity, price, stop_loss, etc.).\n"
    "- If uncertain, prefer HOLD or DEFENSIVE_HOLD.\n"
    "- formula_suggestion is advisory; you may agree or override it.\n\n"
    "Respond with ONLY a JSON object — no markdown, no extra text:\n"
    "{\n"
    '  "action": "<one of the 8 actions>",\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "reason_codes": ["CODE1", "CODE2"],\n'
    '  "why_not": {"HOLD": "<short reason>", "FULL_CLOSE": "<short reason>"},\n'
    '  "risk_note": "<optional, max 200 chars>"\n'
    "}"
)


def _build_payload(ss: "ExitScoreState") -> str:  # type: ignore[name-defined]
    """Serialise ExitScoreState into the structured JSON input for the judge."""
    payload = {
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
    return json.dumps(payload, separators=(",", ":"))


def _needs_second_opinion(result: ValidationResult, ss: "ExitScoreState") -> bool:  # type: ignore[name-defined]
    """
    Soft failure check — determines if primary result needs fallback as second opinion.
    Returns True if any soft-failure condition is met.
    """
    if result.confidence < _CONFIDENCE_THRESHOLD:
        return True
    # High-risk actions on large positions require confirmation
    if result.action in ("FULL_CLOSE", "TOXICITY_UNWIND") and abs(ss.R_net) > 2.0:
        return True
    # Extreme kill chain elevation
    return False


def _action_family(action: str) -> str:
    """Bucket actions into coarse families for disagreement detection."""
    if action in ("HOLD", "DEFENSIVE_HOLD"):
        return "hold"
    if action in ("REDUCE_25", "REDUCE_50", "HARVEST_70_KEEP_30"):
        return "reduce"
    if action in ("FULL_CLOSE", "TOXICITY_UNWIND"):
        return "close"
    if action == "QUARANTINE":
        return "quarantine"
    return "unknown"


def _resolve_disagreement(a1: str, a2: str) -> str:
    """
    Conservative middle-action policy for strong disagreement.
    Falls back to DEFENSIVE_HOLD when no explicit mapping exists.
    """
    middle = _MIDDLE_ACTION.get((a1, a2)) or _MIDDLE_ACTION.get((a2, a1))
    if middle:
        return middle
    f1, f2 = _action_family(a1), _action_family(a2)
    if f1 == f2:
        # Same family — prefer the less aggressive of the two
        family_order = ["hold", "reduce", "close", "quarantine"]
        return a1 if family_order.index(f1) <= family_order.index(f2) else a2
    return _DEFAULT_MIDDLE


@dataclass
class JudgeResult:
    """
    Output of AIJudge.evaluate() for one position.

    tier: "t1"=primary, "t2"=fallback, "t3"=disagreement-resolved, "t0"=formula
    shadow_mode: which rollout phase was active
    """
    action: str
    confidence: float
    reason_codes: tuple
    risk_note: str
    tier: str              # "t1" | "t2" | "t3" | "t0"
    fallback: bool         # True when AI tiers failed and formula took over
    shadow_mode: str       # "shadow" | "hybrid" | "live"
    primary_raw: str       # raw primary response (audit)
    fallback_raw: str      # raw fallback response if called (audit)
    primary_validation: str    # rejection_reason or "" (audit)
    fallback_validation: str   # rejection_reason or "" (audit)
    latency_ms: float
    formula_action: str    # always preserved (audit)

    def as_qwen3_layer_result(self) -> "Qwen3LayerResult":  # type: ignore[name-defined]
        """Returns Qwen3LayerResult for backward compat with main.py wiring."""
        from .models import Qwen3LayerResult
        reason = f"{self.tier}:{','.join(self.reason_codes) or self.risk_note or 'ok'}"
        return Qwen3LayerResult(
            action=self.action,
            confidence=self.confidence,
            reason=reason[:200],
            fallback=self.fallback,
            latency_ms=self.latency_ms,
            raw=self.primary_raw[:500],
        )


class AIJudge:
    """
    3-layer live judge for exit decisions.

    AIJudge.evaluate(score_state) implements the full PATCH-11 failover chain:
      Layer 1 (primary): qwen/qwen3-32b
      Layer 2 (fallback): openai/gpt-oss-20b
      Layer 0 (formula): deterministic safe degrade on double failure

    Never raises. Never blocks the tick loop indefinitely.
    """

    def __init__(
        self,
        primary: GroqModelClient,
        fallback: GroqModelClient,
        shadow_mode: str = "shadow",
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        strict_validation: bool = False,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._shadow_mode = shadow_mode
        self._confidence_threshold = confidence_threshold
        self._strict_validation = strict_validation
        _log.info(
            "[AIJudge] Initialised primary=%s fallback=%s shadow_mode=%s",
            primary.model,
            fallback.model,
            shadow_mode,
        )

    async def evaluate(self, score_state) -> JudgeResult:  # type: ignore[name-defined]
        """
        Evaluate one position. Implements full PATCH-11 failover chain.
        Never raises.
        """
        t0 = time.monotonic()
        user_content = _build_payload(score_state)
        formula_action = score_state.formula_action

        primary_raw = ""
        fallback_raw = ""
        primary_val_reason = ""
        fallback_val_reason = ""

        # ── Layer 1: Primary (Qwen3-32b) ──────────────────────────────────────
        primary_vr: Optional[ValidationResult] = None
        try:
            primary_raw = await self._primary.chat(_SYSTEM_PROMPT, user_content)
            primary_vr = validate(primary_raw, strict_unknown_keys=self._strict_validation)
            primary_val_reason = primary_vr.rejection_reason
        except ThrottleSkipError as exc:
            primary_val_reason = f"throttle:{exc}"
            _log.debug("[AIJudge] Primary throttled: %s", exc)
        except Exception as exc:
            primary_val_reason = f"hard_fail:{type(exc).__name__}"
            _log.warning(
                "[AIJudge] Primary hard failure: %s — escalating to Layer 2",
                exc,
            )

        # ── Layer 2 decision: hard failure OR soft failure? ───────────────────
        needs_fallback = False
        second_opinion_mode = False

        if primary_vr is None or not primary_vr.ok:
            # Hard failure — Layer 2 required
            needs_fallback = True
        elif _needs_second_opinion(primary_vr, score_state):
            # Soft failure — Layer 2 as second opinion
            needs_fallback = True
            second_opinion_mode = True

        fallback_vr: Optional[ValidationResult] = None
        if needs_fallback:
            try:
                fallback_raw = await self._fallback.chat(_SYSTEM_PROMPT, user_content)
                fallback_vr = validate(
                    fallback_raw, strict_unknown_keys=self._strict_validation
                )
                fallback_val_reason = fallback_vr.rejection_reason
            except ThrottleSkipError as exc:
                fallback_val_reason = f"throttle:{exc}"
                _log.debug("[AIJudge] Fallback throttled: %s", exc)
            except Exception as exc:
                fallback_val_reason = f"hard_fail:{type(exc).__name__}"
                _log.warning("[AIJudge] Fallback hard failure: %s", exc)

        latency_ms = (time.monotonic() - t0) * 1000.0

        # ── Result selection ──────────────────────────────────────────────────
        if primary_vr and primary_vr.ok and not needs_fallback:
            # Normal path — primary succeeded, no second opinion needed
            return self._make_result(
                vr=primary_vr,
                tier="t1",
                formula_action=formula_action,
                primary_raw=primary_raw,
                fallback_raw=fallback_raw,
                primary_val=primary_val_reason,
                fallback_val=fallback_val_reason,
                latency_ms=latency_ms,
            )

        if second_opinion_mode and primary_vr and primary_vr.ok:
            if fallback_vr and fallback_vr.ok:
                # Both models responded — check agreement
                if _action_family(primary_vr.action) == _action_family(fallback_vr.action):
                    # Same family — take primary
                    return self._make_result(
                        vr=primary_vr,
                        tier="t1",
                        formula_action=formula_action,
                        primary_raw=primary_raw,
                        fallback_raw=fallback_raw,
                        primary_val=primary_val_reason,
                        fallback_val=fallback_val_reason,
                        latency_ms=latency_ms,
                    )
                else:
                    # Strong disagreement — conservative middle policy
                    middle = _resolve_disagreement(primary_vr.action, fallback_vr.action)
                    _log.info(
                        "[AIJudge] Disagreement primary=%s fallback=%s -> middle=%s",
                        primary_vr.action,
                        fallback_vr.action,
                        middle,
                    )
                    mid_vr = ValidationResult(
                        ok=True,
                        action=middle,
                        confidence=min(primary_vr.confidence, fallback_vr.confidence),
                        reason_codes=primary_vr.reason_codes,
                        risk_note="disagreement_resolved",
                        rejection_reason="",
                        raw_json=primary_vr.raw_json,
                    )
                    return self._make_result(
                        vr=mid_vr,
                        tier="t3",
                        formula_action=formula_action,
                        primary_raw=primary_raw,
                        fallback_raw=fallback_raw,
                        primary_val=primary_val_reason,
                        fallback_val=fallback_val_reason,
                        latency_ms=latency_ms,
                    )
            else:
                # Second opinion failed — proceed with primary alone
                return self._make_result(
                    vr=primary_vr,
                    tier="t1",
                    formula_action=formula_action,
                    primary_raw=primary_raw,
                    fallback_raw=fallback_raw,
                    primary_val=primary_val_reason,
                    fallback_val=fallback_val_reason,
                    latency_ms=latency_ms,
                )

        if fallback_vr and fallback_vr.ok:
            # Hard failure of primary, fallback succeeded
            return self._make_result(
                vr=fallback_vr,
                tier="t2",
                formula_action=formula_action,
                primary_raw=primary_raw,
                fallback_raw=fallback_raw,
                primary_val=primary_val_reason,
                fallback_val=fallback_val_reason,
                latency_ms=latency_ms,
            )

        # ── Layer 0: Double failure — deterministic safe degrade ──────────────
        _log.warning(
            "[AIJudge] DOUBLE_FAILURE primary_err=%r fallback_err=%r "
            "-> safe degrade formula=%s",
            primary_val_reason,
            fallback_val_reason,
            formula_action,
        )
        return JudgeResult(
            action=formula_action,
            confidence=0.0,
            reason_codes=("FORMULA_OVERRIDE",),
            risk_note="double_failure_safe_degrade",
            tier="t0",
            fallback=True,
            shadow_mode=self._shadow_mode,
            primary_raw=primary_raw[:500],
            fallback_raw=fallback_raw[:500],
            primary_validation=primary_val_reason,
            fallback_validation=fallback_val_reason,
            latency_ms=latency_ms,
            formula_action=formula_action,
        )

    def _make_result(
        self,
        vr: ValidationResult,
        tier: str,
        formula_action: str,
        primary_raw: str,
        fallback_raw: str,
        primary_val: str,
        fallback_val: str,
        latency_ms: float,
    ) -> JudgeResult:
        # Apply shadow/hybrid mode restrictions
        final_action = vr.action
        if self._shadow_mode == "shadow":
            # Audit only — formula is live truth (applied in main.py)
            pass  # action kept for audit record; main.py ignores it
        elif self._shadow_mode == "hybrid":
            if final_action not in _HYBRID_ALLOWED:
                # Non-soft action in hybrid mode — fall back to formula
                final_action = formula_action
                _log.debug(
                    "[AIJudge] Hybrid mode: downgraded %r -> formula %r",
                    vr.action,
                    formula_action,
                )

        return JudgeResult(
            action=final_action,
            confidence=vr.confidence,
            reason_codes=vr.reason_codes,
            risk_note=vr.risk_note,
            tier=tier,
            fallback=(tier == "t0"),
            shadow_mode=self._shadow_mode,
            primary_raw=primary_raw[:500],
            fallback_raw=fallback_raw[:500],
            primary_validation=primary_val,
            fallback_validation=fallback_val,
            latency_ms=latency_ms,
            formula_action=formula_action,
        )


# ── Offline evaluator (NEVER on live tick path) ───────────────────────────────

_EVAL_SYSTEM_PROMPT = (
    "You are a post-trade decision auditor for a futures trading system.\n"
    "You receive a JSON record of a completed trade including:\n"
    "  - the action taken (live_action)\n"
    "  - formula suggestion (formula_action)\n"
    "  - AI suggestions if recorded (qwen_action, fallback_action)\n"
    "  - outcome metrics (reward, regret_label, preferred_action, hold_duration_sec)\n\n"
    "Your task: was live_action the best possible decision given what was known?\n\n"
    "Respond with ONLY a JSON object — no markdown, no extra text:\n"
    "{\n"
    '  "was_correct": true/false,\n'
    '  "better_action": "<action from HOLD|REDUCE_25|REDUCE_50|HARVEST_70_KEEP_30|'
    'FULL_CLOSE|DEFENSIVE_HOLD|TOXICITY_UNWIND|QUARANTINE>",\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "reasoning": "<max 200 chars>",\n'
    '  "should_have_exited_earlier": true/false,\n'
    '  "policy_notes": "<max 150 chars>"\n'
    "}"
)

_MAX_EVAL_REASON_LEN = 200


class OfflineEvaluator:
    """
    Post-trade LLM evaluator — replay/obituary/forensic analysis only.

    Uses llama-3.3-70b-versatile via Groq.
    MUST NEVER be called from the live tick path.

    Called exclusively from ReplayWriter.write() after a position closes.
    Returns dict of deepseek_* fields (kept for backward compat with PATCH-11A)
    that are merged into the replay record before Redis xadd.
    """

    def __init__(self, client: GroqModelClient, enabled: bool = True) -> None:
        self._client = client
        self._enabled = enabled
        if enabled:
            _log.info(
                "[OfflineEvaluator] Initialised model=%s", client.model
            )

    async def evaluate_replay(self, record: dict) -> dict:
        """
        Evaluate one closed-trade replay record.

        Must only be called from ReplayWriter — never from the tick loop.
        Returns dict with deepseek_* keys (naming kept for replay schema compat).
        Never raises.
        """
        if not self._enabled:
            return {"deepseek_fallback": "true", "deepseek_skip_reason": "disabled"}

        payload = {
            "live_action": record.get("live_action", "UNKNOWN"),
            "formula_action": record.get("formula_action", "null"),
            "qwen_action": record.get("qwen3_action", "null"),
            "reward": record.get("reward", "null"),
            "regret_label": record.get("regret_label", "null"),
            "preferred_action": record.get("preferred_action", "null"),
            "hold_duration_sec": record.get("hold_duration_sec", "null"),
            "closed_by": record.get("closed_by", "unknown"),
            "symbol": record.get("symbol", "UNKNOWN"),
        }
        user_content = json.dumps(payload, separators=(",", ":"))

        t0 = time.monotonic()
        try:
            raw = await self._client.chat(_EVAL_SYSTEM_PROMPT, user_content)
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            _log.warning(
                "[OfflineEvaluator] HTTP error after %.0fms: %s",
                latency_ms,
                exc,
            )
            return {
                "deepseek_fallback": "true",
                "deepseek_latency_ms": f"{latency_ms:.0f}",
                "deepseek_error": type(exc).__name__[:40],
            }

        latency_ms = (time.monotonic() - t0) * 1000.0
        result = self._parse(raw, latency_ms)
        _log.info(
            "[OfflineEvaluator] symbol=%s was_correct=%s better_action=%s "
            "confidence=%s latency=%.0fms",
            record.get("symbol", "?"),
            result.get("deepseek_was_correct"),
            result.get("deepseek_better_action"),
            result.get("deepseek_confidence"),
            latency_ms,
        )
        return result

    @staticmethod
    def _parse(raw: str, latency_ms: float) -> dict:
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
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
            "deepseek_reasoning": str(parsed.get("reasoning", ""))[:_MAX_EVAL_REASON_LEN],
            "deepseek_should_have_exited_earlier": str(
                parsed.get("should_have_exited_earlier", "unknown")
            ).lower(),
            "deepseek_policy_notes": str(parsed.get("policy_notes", ""))[:150],
            "deepseek_latency_ms": f"{latency_ms:.0f}",
            "deepseek_fallback": "false",
            "deepseek_model": "llama-3.3-70b-versatile",
        }

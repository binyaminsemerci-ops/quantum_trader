"""judge_validator: PATCH-11 — strict JSON schema + policy validator for model output.

Validates both primary and fallback model responses against the shared output
contract before any action is admitted to the gateway.

Approved reason-code registry is deliberately small and explicit — models may
only use codes they have been told about in the system prompt.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("exit_management_agent.judge_validator")

# ── Live action enum (PATCH-11) ───────────────────────────────────────────────
LIVE_ACTIONS: frozenset = frozenset({
    "HOLD",
    "REDUCE_25",
    "REDUCE_50",
    "HARVEST_70_KEEP_30",
    "FULL_CLOSE",
    "DEFENSIVE_HOLD",
    "TOXICITY_UNWIND",
    "QUARANTINE",
})

# DO NOT add FLIP_CANDIDATE — live FLIP is not enabled in PATCH-11.

# ── Approved reason codes ────────────────────────────────────────────────────
APPROVED_REASON_CODES: frozenset = frozenset({
    "THESIS_DECAY",
    "TOXICITY_RISING",
    "PAYOUT_FLATTENING",
    "ALPHA_DEPLETED",
    "CONVICTION_BUDGET_LOW",
    "REGIME_DRIFT",
    "GIVEBACK_EXCESSIVE",
    "SL_PROXIMITY",
    "TIME_LIMIT_APPROACHING",
    "PROFIT_TARGET_REACHED",
    "LOSS_PRESSURE",
    "VELOCITY_NEGATIVE",
    "ACCELERATION_NEGATIVE",
    "COUNCIL_CLOSE",
    "KILL_CHAIN_ELEVATED",
    "NARRATIVE_COLLAPSE",
    "LOW_CONFIDENCE",
    "FORMULA_OVERRIDE",
    "PORTFOLIO_RISK",
})

# ── Text length caps ──────────────────────────────────────────────────────────
_MAX_REASON_CODE_COUNT = 5
_MAX_RISK_NOTE_LEN = 200
_MAX_WHY_NOT_VALUE_LEN = 120
_MAX_WHY_NOT_KEYS = 4

# ── Required fields ───────────────────────────────────────────────────────────
_REQUIRED_FIELDS = {"action", "confidence", "reason_codes"}

# Forbidden keys: LLM must not invent execution parameters
_FORBIDDEN_KEYS: frozenset = frozenset({
    "quantity", "price", "stop_loss", "take_profit", "leverage",
    "order_type", "side", "symbol", "qty_fraction", "sl_price",
    "tp_price", "reduce_only",
})


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    action: str             # validated action (or "" on failure)
    confidence: float       # validated confidence (or 0.0 on failure)
    reason_codes: tuple     # validated codes
    risk_note: str          # sanitised (or "")
    rejection_reason: str   # machine-readable; "" when ok=True
    raw_json: str           # first 500 chars of raw input (audit)


def validate(raw: str, strict_unknown_keys: bool = False) -> ValidationResult:
    """
    Parse and validate a raw model response string against the PATCH-11 contract.

    Returns ValidationResult with ok=True on success, ok=False with
    rejection_reason set on any failure.  Never raises.
    """
    raw_captured = raw[:500] if raw else ""

    def _fail(reason: str) -> ValidationResult:
        _log.debug("[Validator] REJECT reason=%r raw=%r", reason, raw_captured)
        return ValidationResult(
            ok=False,
            action="",
            confidence=0.0,
            reason_codes=(),
            risk_note="",
            rejection_reason=reason,
            raw_json=raw_captured,
        )

    # ── 1. strip DeepSeek/Qwen3 <think> blocks ───────────────────────────────
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    if not cleaned:
        return _fail("empty_response")

    # ── 2. JSON parse ─────────────────────────────────────────────────────────
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        return _fail(f"json_parse_error:{type(exc).__name__}")

    if not isinstance(parsed, dict):
        return _fail("response_not_object")

    # ── 3. Forbidden keys check ───────────────────────────────────────────────
    forbidden = _FORBIDDEN_KEYS & set(parsed.keys())
    if forbidden:
        return _fail(f"forbidden_keys:{','.join(sorted(forbidden))}")

    # ── 4. Unknown keys (strict mode) ────────────────────────────────────────
    if strict_unknown_keys:
        known = {"action", "confidence", "reason_codes", "why_not", "risk_note"}
        unknown = set(parsed.keys()) - known
        if unknown:
            return _fail(f"unknown_keys:{','.join(sorted(unknown))}")

    # ── 5. Required fields ────────────────────────────────────────────────────
    for field in _REQUIRED_FIELDS:
        if field not in parsed:
            return _fail(f"missing_field:{field}")

    # ── 6. Action validation ──────────────────────────────────────────────────
    action = parsed.get("action", "")
    if not isinstance(action, str) or action not in LIVE_ACTIONS:
        return _fail(f"illegal_action:{action!r}")

    # ── 7. Confidence validation ──────────────────────────────────────────────
    try:
        confidence = float(parsed["confidence"])
    except (TypeError, ValueError):
        return _fail("confidence_not_float")
    if not (0.0 <= confidence <= 1.0):
        return _fail(f"confidence_out_of_range:{confidence}")

    # ── 8. reason_codes validation ────────────────────────────────────────────
    raw_codes = parsed.get("reason_codes", [])
    if not isinstance(raw_codes, list):
        return _fail("reason_codes_not_list")
    if len(raw_codes) > _MAX_REASON_CODE_COUNT:
        return _fail(f"reason_codes_too_many:{len(raw_codes)}")
    bad_codes = [c for c in raw_codes if c not in APPROVED_REASON_CODES]
    if bad_codes:
        return _fail(f"unknown_reason_codes:{','.join(str(c) for c in bad_codes[:3])}")
    validated_codes = tuple(str(c) for c in raw_codes)

    # ── 9. why_not (optional) ────────────────────────────────────────────────
    why_not = parsed.get("why_not")
    if why_not is not None:
        if not isinstance(why_not, dict):
            return _fail("why_not_not_object")
        if len(why_not) > _MAX_WHY_NOT_KEYS:
            return _fail(f"why_not_too_many_keys:{len(why_not)}")
        for k, v in why_not.items():
            if k not in LIVE_ACTIONS:
                return _fail(f"why_not_illegal_action_key:{k!r}")
            if not isinstance(v, str) or len(v) > _MAX_WHY_NOT_VALUE_LEN:
                return _fail(f"why_not_value_too_long_or_not_str:{k!r}")

    # ── 10. risk_note (optional) ─────────────────────────────────────────────
    risk_note_raw = parsed.get("risk_note", "")
    if not isinstance(risk_note_raw, str):
        return _fail("risk_note_not_str")
    risk_note = risk_note_raw[:_MAX_RISK_NOTE_LEN]

    return ValidationResult(
        ok=True,
        action=action,
        confidence=confidence,
        reason_codes=validated_codes,
        risk_note=risk_note,
        rejection_reason="",
        raw_json=raw_captured,
    )

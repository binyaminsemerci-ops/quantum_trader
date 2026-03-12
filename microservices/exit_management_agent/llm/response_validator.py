"""PATCH-11 — Strict JSON response validator for LLM judge outputs.

Validates both primary (Qwen3-32b) and fallback (GPT-OSS 20B) responses.

Required schema:
{
  "action": "REDUCE_25",
  "confidence": 0.78,
  "reason_codes": ["THESIS_DECAY", "TOXICITY_RISING"],
  "why_not": {"HOLD": "thesis weakened", "FULL_CLOSE": "edge remains"},
  "risk_note": "Prefer partial de-risk."
}

Validates:
  - Valid JSON
  - All required fields present
  - action in PATCH11_ACTIONS
  - confidence in [0,1]
  - reason_codes all in VALID_REASON_CODES
  - why_not keys must be in PATCH11_ACTIONS
  - No unknown top-level keys (strict mode)
  - Max string lengths
  - No execution parameters

Returns machine-readable rejection reason.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..patch11_actions import PATCH11_ACTIONS, VALID_REASON_CODES

_log = logging.getLogger("exit_management_agent.llm.response_validator")

_REQUIRED_FIELDS = {"action", "confidence", "reason_codes", "why_not", "risk_note"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS  # strict mode: no extra keys
_MAX_REASON_CODES = 10
_MAX_WHY_NOT_ENTRIES = 8
_MAX_STRING_LEN = 300
_MAX_RAW_LEN = 2000
_FORBIDDEN_KEYS = frozenset({
    "quantity", "size", "amount", "price", "order_type", "leverage",
    "stop_loss", "take_profit", "limit_price", "market_order",
    "execution", "order", "position_size",
})


@dataclass(frozen=True)
class ValidatedResponse:
    """Result of LLM response validation."""
    valid: bool
    action: str                    # PATCH-11 action or "" if invalid
    confidence: float              # [0,1] or 0.0 if invalid
    reason_codes: List[str]
    why_not: Dict[str, str]
    risk_note: str
    rejection_reason: Optional[str]  # None if valid; machine-readable code if rejected
    raw: str                       # captured raw response (truncated)


def validate_llm_response(raw: str) -> ValidatedResponse:
    """
    Validate a raw LLM response string against the PATCH-11 schema.

    Never raises. Returns ValidatedResponse with valid=False on any error.
    """
    raw_captured = raw[:_MAX_RAW_LEN] if raw else ""

    # 1. Parse JSON
    try:
        parsed = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return ValidatedResponse(
            valid=False, action="", confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason=f"INVALID_JSON:{type(exc).__name__}",
            raw=raw_captured,
        )

    if not isinstance(parsed, dict):
        return ValidatedResponse(
            valid=False, action="", confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason="NOT_A_JSON_OBJECT",
            raw=raw_captured,
        )

    # 2. Check for forbidden execution parameter keys
    for key in parsed:
        if key in _FORBIDDEN_KEYS:
            return ValidatedResponse(
                valid=False, action="", confidence=0.0,
                reason_codes=[], why_not={}, risk_note="",
                rejection_reason=f"FORBIDDEN_KEY:{key}",
                raw=raw_captured,
            )

    # 3. Check required fields
    missing = _REQUIRED_FIELDS - set(parsed.keys())
    if missing:
        return ValidatedResponse(
            valid=False, action="", confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason=f"MISSING_FIELDS:{','.join(sorted(missing))}",
            raw=raw_captured,
        )

    # 4. Check for unknown keys (strict mode)
    unknown = set(parsed.keys()) - _ALLOWED_FIELDS
    if unknown:
        return ValidatedResponse(
            valid=False, action="", confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason=f"UNKNOWN_KEYS:{','.join(sorted(unknown))}",
            raw=raw_captured,
        )

    # 5. Validate action
    action = parsed.get("action", "")
    if not isinstance(action, str) or action not in PATCH11_ACTIONS:
        return ValidatedResponse(
            valid=False, action="", confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason=f"INVALID_ACTION:{action}",
            raw=raw_captured,
        )

    # 6. Validate confidence
    try:
        confidence = float(parsed.get("confidence", -1))
    except (TypeError, ValueError):
        return ValidatedResponse(
            valid=False, action=action, confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason="INVALID_CONFIDENCE:not_numeric",
            raw=raw_captured,
        )
    if not (0.0 <= confidence <= 1.0):
        return ValidatedResponse(
            valid=False, action=action, confidence=0.0,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason=f"CONFIDENCE_OUT_OF_RANGE:{confidence}",
            raw=raw_captured,
        )

    # 7. Validate reason_codes
    reason_codes_raw = parsed.get("reason_codes", [])
    if not isinstance(reason_codes_raw, list):
        return ValidatedResponse(
            valid=False, action=action, confidence=confidence,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason="REASON_CODES_NOT_ARRAY",
            raw=raw_captured,
        )
    if len(reason_codes_raw) > _MAX_REASON_CODES:
        return ValidatedResponse(
            valid=False, action=action, confidence=confidence,
            reason_codes=[], why_not={}, risk_note="",
            rejection_reason=f"TOO_MANY_REASON_CODES:{len(reason_codes_raw)}",
            raw=raw_captured,
        )
    reason_codes: List[str] = []
    for rc in reason_codes_raw:
        if not isinstance(rc, str):
            return ValidatedResponse(
                valid=False, action=action, confidence=confidence,
                reason_codes=[], why_not={}, risk_note="",
                rejection_reason=f"REASON_CODE_NOT_STRING:{rc}",
                raw=raw_captured,
            )
        if rc not in VALID_REASON_CODES:
            return ValidatedResponse(
                valid=False, action=action, confidence=confidence,
                reason_codes=[], why_not={}, risk_note="",
                rejection_reason=f"UNKNOWN_REASON_CODE:{rc}",
                raw=raw_captured,
            )
        reason_codes.append(rc)

    # 8. Validate why_not
    why_not_raw = parsed.get("why_not", {})
    if not isinstance(why_not_raw, dict):
        return ValidatedResponse(
            valid=False, action=action, confidence=confidence,
            reason_codes=reason_codes, why_not={}, risk_note="",
            rejection_reason="WHY_NOT_NOT_OBJECT",
            raw=raw_captured,
        )
    if len(why_not_raw) > _MAX_WHY_NOT_ENTRIES:
        return ValidatedResponse(
            valid=False, action=action, confidence=confidence,
            reason_codes=reason_codes, why_not={}, risk_note="",
            rejection_reason=f"TOO_MANY_WHY_NOT:{len(why_not_raw)}",
            raw=raw_captured,
        )
    why_not: Dict[str, str] = {}
    for key, val in why_not_raw.items():
        if key not in PATCH11_ACTIONS:
            return ValidatedResponse(
                valid=False, action=action, confidence=confidence,
                reason_codes=reason_codes, why_not={}, risk_note="",
                rejection_reason=f"WHY_NOT_INVALID_ACTION:{key}",
                raw=raw_captured,
            )
        why_not[key] = str(val)[:_MAX_STRING_LEN]

    # 9. Validate risk_note
    risk_note_raw = parsed.get("risk_note", "")
    if not isinstance(risk_note_raw, str):
        risk_note = str(risk_note_raw)[:_MAX_STRING_LEN]
    elif len(risk_note_raw) > _MAX_STRING_LEN:
        return ValidatedResponse(
            valid=False, action=action, confidence=confidence,
            reason_codes=reason_codes, why_not=why_not, risk_note="",
            rejection_reason=f"RISK_NOTE_TOO_LONG:{len(risk_note_raw)}",
            raw=raw_captured,
        )
    else:
        risk_note = risk_note_raw

    # 10. Check for contradictions: empty response
    if not reason_codes and action != "HOLD":
        return ValidatedResponse(
            valid=False, action=action, confidence=confidence,
            reason_codes=reason_codes, why_not=why_not, risk_note=risk_note,
            rejection_reason="NO_REASON_CODES_FOR_NON_HOLD",
            raw=raw_captured,
        )

    return ValidatedResponse(
        valid=True,
        action=action,
        confidence=confidence,
        reason_codes=reason_codes,
        why_not=why_not,
        risk_note=risk_note,
        rejection_reason=None,
        raw=raw_captured,
    )

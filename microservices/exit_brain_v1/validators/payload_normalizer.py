"""
PayloadNormalizer — Clamps, strips, and validates intent payloads.

Pure logic. No IO. No state.
Used by: exit_intent_gateway_validator.py
"""

from __future__ import annotations

from typing import Dict, List

from ..models.action_candidate import ACTION_EXIT_FRACTIONS, VALID_ACTIONS


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# Fields allowed in the normalized payload
_ALLOWED_FIELDS = frozenset({
    "position_id", "symbol", "intent_id", "intent_timestamp",
    "action_name", "intent_type", "target_reduction_pct",
    "tighten_parameters", "justification_summary", "source_decision_id",
    "confidence", "uncertainty", "constraint_flags", "quality_flags",
    "idempotency_key", "shadow_only",
})

# Required fields that must be present and non-empty
_REQUIRED_FIELDS = frozenset({
    "position_id", "symbol", "intent_id", "action_name",
    "intent_type", "source_decision_id", "idempotency_key",
})


def normalize_payload(raw: Dict) -> tuple[Dict, List[str]]:
    """
    Normalize an intent candidate payload.

    - Strip unexpected fields
    - Clamp numeric ranges
    - Ensure required fields exist

    Returns:
        (normalized_dict, list_of_warnings)
    """
    warnings: List[str] = []

    # Strip unexpected fields
    normalized = {k: v for k, v in raw.items() if k in _ALLOWED_FIELDS}
    stripped = set(raw.keys()) - _ALLOWED_FIELDS
    if stripped:
        warnings.append(f"stripped_fields={sorted(stripped)}")

    # Clamp percentages
    if "target_reduction_pct" in normalized:
        val = normalized["target_reduction_pct"]
        if isinstance(val, (int, float)):
            normalized["target_reduction_pct"] = _clamp(float(val))

    if "confidence" in normalized:
        val = normalized["confidence"]
        if isinstance(val, (int, float)):
            normalized["confidence"] = _clamp(float(val))

    if "uncertainty" in normalized:
        val = normalized["uncertainty"]
        if isinstance(val, (int, float)):
            normalized["uncertainty"] = _clamp(float(val))

    # Validate action name
    action = normalized.get("action_name", "")
    if action and action not in VALID_ACTIONS:
        warnings.append(f"invalid_action={action}")

    # Validate intent type
    intent_type = normalized.get("intent_type", "")
    if intent_type and intent_type != "SHADOW_EXIT":
        warnings.append(f"unexpected_intent_type={intent_type}")

    # Validate reduction_pct matches action
    expected_frac = ACTION_EXIT_FRACTIONS.get(action, 0.0)
    actual_frac = normalized.get("target_reduction_pct", 0.0)
    if isinstance(actual_frac, (int, float)) and abs(actual_frac - expected_frac) > 0.01:
        warnings.append(
            f"reduction_pct_mismatch: expected={expected_frac}, got={actual_frac}"
        )

    # Force shadow_only
    normalized["shadow_only"] = True

    return normalized, warnings


def check_required_fields(payload: Dict) -> List[str]:
    """
    Check that all required fields are present and non-empty.

    Returns list of missing/empty field names.
    """
    missing: List[str] = []
    for field_name in _REQUIRED_FIELDS:
        val = payload.get(field_name)
        if val is None or (isinstance(val, str) and not val):
            missing.append(field_name)
    return missing

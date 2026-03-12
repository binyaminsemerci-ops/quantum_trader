"""validator: pre-publish validation for exit intents (PATCH-5A).

ExitIntentValidator performs synchronous, Redis-free checks on an ExitDecision
before IntentWriter is allowed to publish it to quantum:stream:exit.intent.

All checks are cheap (no I/O).  The validator is called inside the hot tick
loop, so it must complete in microseconds.

Validation rules (7 checks, evaluated in order — first failure short-circuits):
  V1  action_whitelisted   — action must be in LIVE_ACTION_WHITELIST
  V2  action_not_hold      — HOLD is never published (redundant safety check)
  V3  urgency_sufficient   — urgency must be MEDIUM, HIGH, or EMERGENCY
  V4  confidence_sufficient— confidence must be >= MIN_CONFIDENCE (0.65)
  V5  data_fresh           — sync_timestamp must be within MAX_DATA_AGE_SEC (30 s)
  V6  notional_sufficient  — mark_price * quantity must be >= MIN_NOTIONAL_USDT (20)
  V7  qty_fraction_valid   — if qty_fraction is set it must be in (0, 1]

Blocked actions (must NEVER reach exit.intent):
  HOLD, TIGHTEN_TRAIL, MOVE_TO_BREAKEVEN
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .models import ExitDecision

# ── Constants ─────────────────────────────────────────────────────────────────

# Only these actions may be published to exit.intent in PATCH-5A v1.
# SL-modification actions (TIGHTEN_TRAIL, MOVE_TO_BREAKEVEN) are intentionally
# excluded — they require a different execution path not yet implemented.
LIVE_ACTION_WHITELIST: frozenset = frozenset(
    {
        "FULL_CLOSE",
        "PARTIAL_CLOSE_25",
        "TIME_STOP_EXIT",
        # PATCH-11 LLM judge actions
        "REDUCE_25",
        "REDUCE_50",
        "HARVEST_70_KEEP_30",
        "TOXICITY_UNWIND",
    }
)

# Urgency levels that are considered actionable enough for live publication.
_SUFFICIENT_URGENCY: frozenset = frozenset({"MEDIUM", "HIGH", "EMERGENCY"})

# Minimum AI/decision confidence to publish.
MIN_CONFIDENCE: float = 0.65

# Maximum age of position data (sync_timestamp) before we consider it stale.
MAX_DATA_AGE_SEC: float = 30.0

# Minimum USD notional (mark_price × quantity) — avoids sending micro orders.
MIN_NOTIONAL_USDT: float = 20.0


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    rule: str            # name of the check that was evaluated last
    reason: str          # human-readable explanation
    decision_action: str # action from the decision being validated

    @property
    def failed(self) -> bool:
        return not self.passed

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({status} rule={self.rule!r} "
            f"action={self.decision_action!r} reason={self.reason!r})"
        )


# ── Validator ─────────────────────────────────────────────────────────────────

class ExitIntentValidator:
    """
    Synchronous, Redis-free validator for ExitDecision objects.

    Usage::

        validator = ExitIntentValidator()
        result = validator.validate(dec)
        if result.passed:
            # safe to publish
    """

    def validate(self, dec: ExitDecision) -> ValidationResult:
        """
        Run all 7 validation checks against *dec*.

        Returns a ValidationResult with passed=True only if every check passes.
        On the first failure, returns immediately with passed=False and a
        descriptive reason.
        """
        action = dec.action

        # V1 — action must be in the live whitelist
        if action not in LIVE_ACTION_WHITELIST:
            return ValidationResult(
                passed=False,
                rule="action_whitelisted",
                reason=(
                    f"Action {action!r} is not in LIVE_ACTION_WHITELIST "
                    f"{sorted(LIVE_ACTION_WHITELIST)}. "
                    "HOLD, TIGHTEN_TRAIL, and MOVE_TO_BREAKEVEN are never published."
                ),
                decision_action=action,
            )

        # V2 — redundant HOLD guard (belt-and-suspenders)
        if action == "HOLD":
            return ValidationResult(
                passed=False,
                rule="action_not_hold",
                reason="HOLD decisions are never published to exit.intent.",
                decision_action=action,
            )

        # V3 — urgency must be MEDIUM, HIGH, or EMERGENCY
        if dec.urgency not in _SUFFICIENT_URGENCY:
            return ValidationResult(
                passed=False,
                rule="urgency_sufficient",
                reason=(
                    f"Urgency {dec.urgency!r} is below threshold. "
                    f"Minimum required: one of {sorted(_SUFFICIENT_URGENCY)}."
                ),
                decision_action=action,
            )

        # V4 — confidence must be >= MIN_CONFIDENCE
        if dec.confidence < MIN_CONFIDENCE:
            return ValidationResult(
                passed=False,
                rule="confidence_sufficient",
                reason=(
                    f"Confidence {dec.confidence:.3f} is below minimum "
                    f"{MIN_CONFIDENCE:.3f}."
                ),
                decision_action=action,
            )

        # V5 — position data must be fresh
        data_age_sec = time.time() - dec.snapshot.sync_timestamp
        if data_age_sec > MAX_DATA_AGE_SEC:
            return ValidationResult(
                passed=False,
                rule="data_fresh",
                reason=(
                    f"Position data is {data_age_sec:.1f}s old "
                    f"(max allowed: {MAX_DATA_AGE_SEC:.0f}s). "
                    "Stale data rejected to prevent acting on outdated prices."
                ),
                decision_action=action,
            )

        # V6 — notional must be >= MIN_NOTIONAL_USDT
        notional = dec.snapshot.mark_price * dec.snapshot.quantity
        if notional < MIN_NOTIONAL_USDT:
            return ValidationResult(
                passed=False,
                rule="notional_sufficient",
                reason=(
                    f"Notional {notional:.2f} USDT is below minimum "
                    f"{MIN_NOTIONAL_USDT:.2f} USDT."
                ),
                decision_action=action,
            )

        # V7 — qty_fraction must be in (0, 1] if provided
        if dec.suggested_qty_fraction is not None:
            frac = dec.suggested_qty_fraction
            if not (0.0 < frac <= 1.0):
                return ValidationResult(
                    passed=False,
                    rule="qty_fraction_valid",
                    reason=(
                        f"suggested_qty_fraction {frac!r} is outside (0, 1]. "
                        "Must be a positive fraction not exceeding 1.0."
                    ),
                    decision_action=action,
                )

        return ValidationResult(
            passed=True,
            rule="all_checks_passed",
            reason="All 7 validation checks passed.",
            decision_action=action,
        )

"""
Reason codes and explanation tags for Exit Policy Engine.

Machine-readable constants. No IO. No state.
"""

from __future__ import annotations

# ── Hard block reason codes (action MUST be overridden to HOLD) ──────────

UNCERTAINTY_CEILING_BREACH = "UNCERTAINTY_CEILING_BREACH"
DATA_COMPLETENESS_FLOOR = "DATA_COMPLETENESS_FLOOR"
PROFIT_TAKING_NO_PROFIT = "PROFIT_TAKING_NO_PROFIT"
CLOSE_FULL_INSUFFICIENT_HAZARD = "CLOSE_FULL_INSUFFICIENT_HAZARD"
STALE_UPSTREAM_DATA = "STALE_UPSTREAM_DATA"
MISSING_UPSTREAM_DATA = "MISSING_UPSTREAM_DATA"
SHADOW_ONLY_VIOLATION = "SHADOW_ONLY_VIOLATION"

# ── Soft warning codes (action proceeds but flagged) ─────────────────────

INSUFFICIENT_CONVICTION = "INSUFFICIENT_CONVICTION"
EDGE_NEUTRAL_HOLD_PREFERRED = "EDGE_NEUTRAL_HOLD_PREFERRED"
HIGH_UNCERTAINTY_DAMPENING = "HIGH_UNCERTAINTY_DAMPENING"
LOW_DATA_COMPLETENESS = "LOW_DATA_COMPLETENESS"
QUALITY_FLAGS_PRESENT = "QUALITY_FLAGS_PRESENT"
PARTIAL_UPSTREAM = "PARTIAL_UPSTREAM"

# ── Policy override codes ────────────────────────────────────────────────

HAZARD_EMERGENCY_OVERRIDE = "HAZARD_EMERGENCY_OVERRIDE"
HAZARD_CLOSE_BOOST = "HAZARD_CLOSE_BOOST"
HOLD_PROMOTED_LOW_CONVICTION = "HOLD_PROMOTED_LOW_CONVICTION"
HOLD_PROMOTED_EDGE_NEUTRAL = "HOLD_PROMOTED_EDGE_NEUTRAL"
POLICY_FALLBACK_HOLD = "POLICY_FALLBACK_HOLD"

# ── Explanation tags (human-readable) ────────────────────────────────────

TAG_SAFE_HOLD = "Safe to hold: low hazard, positive edge"
TAG_UNCERTAIN = "High uncertainty dampens aggressive actions"
TAG_EMERGENCY_EXIT = "Emergency exit: extreme hazard override"
TAG_PROFIT_LOCK = "Locking profits: high drawdown risk"
TAG_RISK_TRIM = "Trimming exposure: moderate risk detected"
TAG_DEFENSIVE = "Defensive tightening: volatility concern"
TAG_CONVICTION_EXIT = "High-conviction exit: ensemble + regime agree"
TAG_INSUFFICIENT_DATA = "Insufficient data: defaulting to HOLD"
TAG_STALE_DATA = "Stale upstream data: defaulting to HOLD"
TAG_DUPLICATE_SUPPRESSED = "Duplicate intent suppressed"
TAG_POLICY_OVERRIDE = "Policy overrode utility ranking"

# ── Convenience sets ─────────────────────────────────────────────────────

ALL_HARD_BLOCKS = frozenset({
    UNCERTAINTY_CEILING_BREACH,
    DATA_COMPLETENESS_FLOOR,
    PROFIT_TAKING_NO_PROFIT,
    CLOSE_FULL_INSUFFICIENT_HAZARD,
    STALE_UPSTREAM_DATA,
    MISSING_UPSTREAM_DATA,
    SHADOW_ONLY_VIOLATION,
})

ALL_SOFT_WARNINGS = frozenset({
    INSUFFICIENT_CONVICTION,
    EDGE_NEUTRAL_HOLD_PREFERRED,
    HIGH_UNCERTAINTY_DAMPENING,
    LOW_DATA_COMPLETENESS,
    QUALITY_FLAGS_PRESENT,
    PARTIAL_UPSTREAM,
})

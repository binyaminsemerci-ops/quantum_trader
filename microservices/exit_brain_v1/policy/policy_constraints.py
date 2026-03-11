"""
Policy constraints — Hard blocks, soft warnings, thresholds.

Pure constants and rule definitions. No IO. No state.
Used by: exit_policy_engine.py
"""

from __future__ import annotations


# ── Uncertainty thresholds ───────────────────────────────────────────────

# Above this, only HOLD and TIGHTEN_EXIT are allowed
UNCERTAINTY_HARD_CEILING = 0.70

# Above this, soft warning applied
UNCERTAINTY_SOFT_CEILING = 0.50

# ── Data completeness thresholds ─────────────────────────────────────────

# Below this, force HOLD (insufficient evidence)
DATA_COMPLETENESS_HARD_FLOOR = 0.40

# Below this, soft warning
DATA_COMPLETENESS_SOFT_FLOOR = 0.60

# ── Conviction thresholds ────────────────────────────────────────────────

# If best non-HOLD candidate has net_utility below this, demote to HOLD
MIN_ACTION_CONVICTION = 0.15

# Stronger threshold: prefer HOLD unless conviction exceeds this
PREFER_HOLD_THRESHOLD = 0.20

# ── Directional edge thresholds ──────────────────────────────────────────

# If directional_edge is within this band, market is edge-neutral → HOLD bias
EDGE_NEUTRAL_BAND = 0.15

# ── Hazard thresholds ───────────────────────────────────────────────────

# Emergency override: force CLOSE_FULL consideration
HAZARD_EMERGENCY_THRESHOLD = 0.85

# CLOSE_FULL requires at least this composite_hazard (or exit_pressure)
CLOSE_FULL_MIN_HAZARD = 0.50
CLOSE_FULL_MIN_EXIT_PRESSURE = 0.70

# Emergency close also triggered by specific sub-hazards
REVERSAL_EMERGENCY_THRESHOLD = 0.70
DRAWDOWN_EMERGENCY_THRESHOLD = 0.50

# ── Hazard boost for CLOSE_FULL ─────────────────────────────────────────

# When composite_hazard > this, CLOSE_FULL gets a policy utility boost
HAZARD_CLOSE_BOOST_THRESHOLD = 0.65
HAZARD_CLOSE_BOOST_AMOUNT = 0.15

# ── Upstream freshness ──────────────────────────────────────────────────

# Maximum age of upstream data before forcing HOLD
MAX_UPSTREAM_AGE_SEC = 120.0

# ── Actions allowed under high uncertainty ───────────────────────────────

SAFE_ACTIONS_HIGH_UNCERTAINTY = frozenset({"HOLD", "TIGHTEN_EXIT"})

# ── Actions that require positive PnL ────────────────────────────────────

PROFIT_REQUIRED_ACTIONS = frozenset({"TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE"})

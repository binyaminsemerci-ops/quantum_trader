"""PATCH-11 — Action enum, reason codes, qty fractions, action families.

Constants only. No IO. No state. No side effects.
Used by: llm/judge_orchestrator, llm/response_validator, llm/disagreement_resolver,
         main.py (PATCH-11 tick path), validator.py (whitelist expansion).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

# ── PATCH-11 live action enum (8 actions, no FLIP) ─────────────────────────

PATCH11_ACTIONS: FrozenSet[str] = frozenset({
    "HOLD",
    "REDUCE_25",
    "REDUCE_50",
    "HARVEST_70_KEEP_30",
    "FULL_CLOSE",
    "DEFENSIVE_HOLD",
    "TOXICITY_UNWIND",
    "QUARANTINE",
})

# ── Qty fraction for each action ───────────────────────────────────────────

PATCH11_QTY_MAP: Dict[str, Optional[float]] = {
    "HOLD": None,
    "REDUCE_25": 0.25,
    "REDUCE_50": 0.50,
    "HARVEST_70_KEEP_30": 0.70,
    "FULL_CLOSE": 1.0,
    "DEFENSIVE_HOLD": None,
    "TOXICITY_UNWIND": 1.0,
    "QUARANTINE": None,
}

# ── Action families for disagreement resolution ────────────────────────────

ACTION_FAMILIES: Dict[str, FrozenSet[str]] = {
    "HOLD_FAMILY": frozenset({"HOLD", "DEFENSIVE_HOLD"}),
    "REDUCE_FAMILY": frozenset({"REDUCE_25", "REDUCE_50"}),
    "HARVEST_FAMILY": frozenset({"HARVEST_70_KEEP_30"}),
    "EXIT_FAMILY": frozenset({"FULL_CLOSE", "TOXICITY_UNWIND"}),
    "QUARANTINE_FAMILY": frozenset({"QUARANTINE"}),
}

ACTION_TO_FAMILY: Dict[str, str] = {}
for _family_name, _actions in ACTION_FAMILIES.items():
    for _action in _actions:
        ACTION_TO_FAMILY[_action] = _family_name

# ── Valid reason codes for LLM responses ───────────────────────────────────

VALID_REASON_CODES: FrozenSet[str] = frozenset({
    "THESIS_DECAY",
    "THESIS_EXHAUSTION",
    "REALITY_DRIFT",
    "TOXICITY_RISING",
    "TOXICITY_CRITICAL",
    "PAYOUT_FLATTENING",
    "GIVEBACK_RISK",
    "REGIME_HOSTILE",
    "REGIME_SHIFT",
    "TIME_DECAY",
    "MOMENTUM_LOSS",
    "REVERSAL_SIGNAL",
    "DRAWDOWN_RISK",
    "VOLATILITY_SPIKE",
    "EDGE_REMAINING",
    "CONVICTION_HIGH",
    "CONVICTION_LOW",
    "PROFIT_LOCKED",
    "UNCERTAINTY_HIGH",
    "ENSEMBLE_CONSENSUS",
    "ENSEMBLE_DISAGREEMENT",
    "CROSS_CHECK_PASS",
    "CROSS_CHECK_FAIL",
})

# ── Map EB v1 actions → closest PATCH-11 action ───────────────────────────

EBV1_TO_PATCH11: Dict[str, str] = {
    "HOLD": "HOLD",
    "REDUCE_SMALL": "REDUCE_25",
    "REDUCE_MEDIUM": "REDUCE_25",
    "TAKE_PROFIT_PARTIAL": "REDUCE_50",
    "TAKE_PROFIT_LARGE": "HARVEST_70_KEEP_30",
    "TIGHTEN_EXIT": "DEFENSIVE_HOLD",
    "CLOSE_FULL": "FULL_CLOSE",
}

# ── Thresholds (defaults — overridden by config) ──────────────────────────

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60
DEFAULT_CONFLICT_THRESHOLD: float = 0.40
DEFAULT_LARGE_POSITION_USDT: float = 5000.0
DEFAULT_HIGH_TOXICITY: float = 0.70

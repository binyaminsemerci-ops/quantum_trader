"""
ActionCandidate — Scored exit action for one position.

Phase 3 data contract. Shadow-only. No execution writes.

Produced by: action_utility_engine.py
Consumed by: (future) policy layer
Published to: quantum:stream:exit.utility.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({
    "HOLD",
    "REDUCE_SMALL",
    "REDUCE_MEDIUM",
    "TAKE_PROFIT_PARTIAL",
    "TAKE_PROFIT_LARGE",
    "TIGHTEN_EXIT",
    "CLOSE_FULL",
})

# What fraction of the position each action implies exiting
ACTION_EXIT_FRACTIONS: Dict[str, float] = {
    "HOLD": 0.00,
    "REDUCE_SMALL": 0.10,
    "REDUCE_MEDIUM": 0.25,
    "TAKE_PROFIT_PARTIAL": 0.50,
    "TAKE_PROFIT_LARGE": 0.75,
    "TIGHTEN_EXIT": 0.00,       # risk management, no immediate exit
    "CLOSE_FULL": 1.00,
}


@dataclass(frozen=False)
class ActionCandidate:
    """
    Scored action candidate with utility breakdown.

    net_utility = clamp(base_utility - penalty_total, 0, 1).
    rank 1 = highest net_utility.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str

    # ── Action (REQUIRED) ────────────────────────────────────────────────
    action: str                        # One of VALID_ACTIONS
    exit_fraction: float               # [0, 1] from ACTION_EXIT_FRACTIONS

    # ── Utility (REQUIRED) ───────────────────────────────────────────────
    base_utility: float                # [0, 1] Raw utility before penalties
    penalty_total: float               # [0, 1] Sum of all penalties
    net_utility: float                 # [0, 1] base - penalty, clamped

    # ── Ranking (REQUIRED) ───────────────────────────────────────────────
    rank: int                          # 1 = best action

    # ── Traceability (REQUIRED) ──────────────────────────────────────────
    utility_components: Dict[str, float] = field(default_factory=dict)
    penalty_components: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    # ── Timing (REQUIRED) ────────────────────────────────────────────────
    scoring_timestamp: float = 0.0

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.position_id:
            errors.append("position_id is empty")
        if not self.symbol:
            errors.append("symbol is empty")
        if self.action not in VALID_ACTIONS:
            errors.append(f"action '{self.action}' not in {VALID_ACTIONS}")

        for name, val in [
            ("exit_fraction", self.exit_fraction),
            ("base_utility", self.base_utility),
            ("penalty_total", self.penalty_total),
            ("net_utility", self.net_utility),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        if self.rank < 1:
            errors.append(f"rank must be >= 1, got {self.rank}")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 3")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "action": self.action,
            "exit_fraction": self.exit_fraction,
            "base_utility": self.base_utility,
            "penalty_total": self.penalty_total,
            "net_utility": self.net_utility,
            "rank": self.rank,
            "utility_components": json.dumps(self.utility_components),
            "penalty_components": json.dumps(self.penalty_components),
            "rationale": self.rationale,
            "scoring_timestamp": self.scoring_timestamp,
            "shadow_only": self.shadow_only,
        }

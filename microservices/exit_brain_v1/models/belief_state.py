"""
BeliefState — Fused situation assessment for one position.

Phase 3 data contract. Shadow-only. Fail-closed.

Produced by: belief_engine.py
Consumed by: hazard_engine.py, action_utility_engine.py
Published to: quantum:stream:exit.belief.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class BeliefState:
    """
    Unified belief about a position's situation.

    Fuses geometry, regime, and ensemble signals into a single
    assessment. All fields are [0,1] or [-1,+1] as documented.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    side: str                          # "LONG" or "SHORT"

    # ── Core beliefs (REQUIRED) ──────────────────────────────────────────
    exit_pressure: float               # [0,1] Composite urgency to exit
    hold_conviction: float             # [0,1] Composite conviction to hold
    directional_edge: float            # [-1,+1] Net directional advantage
    uncertainty_total: float           # [0,1] Combined system uncertainty
    data_completeness: float           # [0,1] Fraction of available signal

    # ── Timing (REQUIRED) ────────────────────────────────────────────────
    belief_timestamp: float            # Epoch seconds

    # ── Traceability (REQUIRED) ──────────────────────────────────────────
    belief_components: Dict[str, float] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)

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
        if self.side not in ("LONG", "SHORT"):
            errors.append(f"side '{self.side}' not in (LONG, SHORT)")

        for name, val, lo, hi in [
            ("exit_pressure", self.exit_pressure, 0.0, 1.0),
            ("hold_conviction", self.hold_conviction, 0.0, 1.0),
            ("directional_edge", self.directional_edge, -1.0, 1.0),
            ("uncertainty_total", self.uncertainty_total, 0.0, 1.0),
            ("data_completeness", self.data_completeness, 0.0, 1.0),
        ]:
            if not (lo <= val <= hi):
                errors.append(f"{name} must be in [{lo}, {hi}], got {val}")

        if self.belief_timestamp <= 0:
            errors.append(f"belief_timestamp must be > 0, got {self.belief_timestamp}")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 3")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "exit_pressure": self.exit_pressure,
            "hold_conviction": self.hold_conviction,
            "directional_edge": self.directional_edge,
            "uncertainty_total": self.uncertainty_total,
            "data_completeness": self.data_completeness,
            "belief_timestamp": self.belief_timestamp,
            "belief_components": json.dumps(self.belief_components),
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }

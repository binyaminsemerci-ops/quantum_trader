"""
HazardAssessment — Multi-dimensional risk assessment for one position.

Phase 3 data contract. Shadow-only. Fail-closed.

Produced by: hazard_engine.py
Consumed by: action_utility_engine.py
Published to: quantum:stream:exit.hazard.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class HazardAssessment:
    """
    Multi-dimensional risk view of a position.

    Six independent hazard axes + one composite.
    All hazard values are in [0, 1] where 0 = no risk, 1 = maximum risk.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str

    # ── Hazard axes (REQUIRED, all [0,1]) ────────────────────────────────
    drawdown_hazard: float             # Profit giveback risk
    reversal_hazard: float             # Market reversal risk
    volatility_hazard: float           # Volatility-driven risk
    time_decay_hazard: float           # Holding duration risk
    regime_hazard: float               # Unfavorable regime risk
    ensemble_hazard: float             # Direct ensemble sell signal

    # ── Composite (REQUIRED) ─────────────────────────────────────────────
    composite_hazard: float            # [0,1] Weighted combination
    dominant_hazard: str               # Name of the highest hazard axis

    # ── Timing (REQUIRED) ────────────────────────────────────────────────
    hazard_timestamp: float            # Epoch seconds

    # ── Traceability (REQUIRED) ──────────────────────────────────────────
    hazard_components: Dict[str, float] = field(default_factory=dict)
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

        for name, val in [
            ("drawdown_hazard", self.drawdown_hazard),
            ("reversal_hazard", self.reversal_hazard),
            ("volatility_hazard", self.volatility_hazard),
            ("time_decay_hazard", self.time_decay_hazard),
            ("regime_hazard", self.regime_hazard),
            ("ensemble_hazard", self.ensemble_hazard),
            ("composite_hazard", self.composite_hazard),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        if self.hazard_timestamp <= 0:
            errors.append(f"hazard_timestamp must be > 0, got {self.hazard_timestamp}")

        if not self.dominant_hazard:
            errors.append("dominant_hazard is empty")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 3")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "drawdown_hazard": self.drawdown_hazard,
            "reversal_hazard": self.reversal_hazard,
            "volatility_hazard": self.volatility_hazard,
            "time_decay_hazard": self.time_decay_hazard,
            "regime_hazard": self.regime_hazard,
            "ensemble_hazard": self.ensemble_hazard,
            "composite_hazard": self.composite_hazard,
            "dominant_hazard": self.dominant_hazard,
            "hazard_timestamp": self.hazard_timestamp,
            "hazard_components": json.dumps(self.hazard_components),
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }

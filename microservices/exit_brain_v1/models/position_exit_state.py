"""
PositionExitState — Canonical data contract for Exit Brain v1.

This is the single source of truth for all position+market state
consumed by downstream exit engines, scorers, and (later) agents.

Rules:
- shadow_only MUST be True in Phase 1. Hard-coded, not configurable.
- Fail-closed: missing REQUIRED fields → builder returns None.
- OPTIONAL fields default to safe values and are flagged in data_quality_flags.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# Valid regime labels (superset of meta_regime + MarketState)
VALID_REGIMES = frozenset({
    "BULL", "BEAR", "RANGE", "VOLATILE", "UNCERTAIN", "UNKNOWN",
    "TREND", "MR", "CHOP",
})

VALID_SIDES = frozenset({"LONG", "SHORT"})
VALID_STATUSES = frozenset({"OPEN", "REDUCING", "CLOSING"})


@dataclass(frozen=False)
class PositionExitState:
    """
    Immutable-ish enriched position state for exit decision-making.

    Produced by: position_state_builder.py
    Consumed by: geometry_engine, regime_drift_engine, (later) belief/utility/policy
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    status: Literal["OPEN", "REDUCING", "CLOSING"]

    # ── Price / Size (REQUIRED) ──────────────────────────────────────────
    entry_price: float
    current_price: float
    quantity: float          # Absolute value, always > 0
    notional: float          # current_price * quantity

    # ── PnL (REQUIRED) ──────────────────────────────────────────────────
    unrealized_pnl: float
    unrealized_pnl_pct: float

    # ── Timestamps (REQUIRED) ────────────────────────────────────────────
    open_timestamp: float    # Epoch seconds
    source_timestamps: Dict[str, float]  # {"p33_snapshot": epoch, "meta_regime": epoch, ...}

    # ── Data quality (REQUIRED) ──────────────────────────────────────────
    data_quality_flags: List[str] = field(default_factory=list)
    shadow_only: bool = True  # MUST be True in Phase 1

    # ── Price (OPTIONAL) ─────────────────────────────────────────────────
    mark_price: float = 0.0

    # ── Leverage (OPTIONAL) ──────────────────────────────────────────────
    leverage: float = 1.0

    # ── PnL tracking (OPTIONAL) ──────────────────────────────────────────
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    # ── Excursion tracking (OPTIONAL) ────────────────────────────────────
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    peak_unrealized_pnl: float = 0.0
    trough_unrealized_pnl: float = 0.0
    drawdown_from_peak_pnl: float = 0.0

    # ── Volatility / Market (OPTIONAL) ───────────────────────────────────
    volatility_short: Optional[float] = None   # sigma from 64-window
    volatility_medium: Optional[float] = None  # sigma from 256-window
    atr: Optional[float] = None

    # ── Regime (OPTIONAL) ────────────────────────────────────────────────
    trend_signal: Optional[float] = None       # mu from MarketState
    regime_label: str = "UNKNOWN"
    regime_confidence: float = 0.0

    # ── Scores (OPTIONAL — populated by engines) ─────────────────────────
    momentum_score: Optional[float] = None
    mean_reversion_score: Optional[float] = None
    liquidity_score: Optional[float] = None
    spread_bps: Optional[float] = None

    # ── Computed ─────────────────────────────────────────────────────────

    @property
    def hold_seconds(self) -> float:
        """Seconds since position opened."""
        return max(0.0, time.time() - self.open_timestamp)

    @property
    def feature_freshness_seconds(self) -> float:
        """Age of the oldest source timestamp (worst-case staleness)."""
        if not self.source_timestamps:
            return float("inf")
        now = time.time()
        return max(now - ts for ts in self.source_timestamps.values())

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """
        Run validation checks. Returns list of error strings.
        Empty list = valid.
        """
        errors: List[str] = []

        # Required field checks
        if not self.position_id:
            errors.append("position_id is empty")
        if not self.symbol:
            errors.append("symbol is empty")
        if self.side not in VALID_SIDES:
            errors.append(f"side '{self.side}' not in {VALID_SIDES}")
        if self.status not in VALID_STATUSES:
            errors.append(f"status '{self.status}' not in {VALID_STATUSES}")
        if self.entry_price <= 0:
            errors.append(f"entry_price must be > 0, got {self.entry_price}")
        if self.current_price <= 0:
            errors.append(f"current_price must be > 0, got {self.current_price}")
        if self.quantity <= 0:
            errors.append(f"quantity must be > 0, got {self.quantity}")
        if self.notional <= 0:
            errors.append(f"notional must be > 0, got {self.notional}")
        if self.open_timestamp <= 0:
            errors.append(f"open_timestamp must be > 0, got {self.open_timestamp}")

        # Shadow enforcement
        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 1")

        # Optional field range checks (non-fatal, add to quality flags)
        if self.leverage < 1.0:
            self.data_quality_flags.append("LEVERAGE_BELOW_1")
        if self.regime_label not in VALID_REGIMES:
            self.data_quality_flags.append(f"INVALID_REGIME:{self.regime_label}")
            self.regime_label = "UNKNOWN"
        if not (0.0 <= self.regime_confidence <= 1.0):
            self.data_quality_flags.append("REGIME_CONFIDENCE_OUT_OF_RANGE")
            self.regime_confidence = max(0.0, min(1.0, self.regime_confidence))

        return errors

    def to_dict(self) -> Dict:
        """Serialize to dict for Redis/JSON publishing."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "mark_price": self.mark_price,
            "quantity": self.quantity,
            "notional": self.notional,
            "leverage": self.leverage,
            "open_timestamp": self.open_timestamp,
            "hold_seconds": self.hold_seconds,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "fees_paid": self.fees_paid,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "peak_unrealized_pnl": self.peak_unrealized_pnl,
            "trough_unrealized_pnl": self.trough_unrealized_pnl,
            "drawdown_from_peak_pnl": self.drawdown_from_peak_pnl,
            "volatility_short": self.volatility_short,
            "volatility_medium": self.volatility_medium,
            "atr": self.atr,
            "trend_signal": self.trend_signal,
            "regime_label": self.regime_label,
            "regime_confidence": self.regime_confidence,
            "momentum_score": self.momentum_score,
            "mean_reversion_score": self.mean_reversion_score,
            "liquidity_score": self.liquidity_score,
            "spread_bps": self.spread_bps,
            "source_timestamps": self.source_timestamps,
            "feature_freshness_seconds": self.feature_freshness_seconds,
            "data_quality_flags": self.data_quality_flags,
            "shadow_only": self.shadow_only,
        }

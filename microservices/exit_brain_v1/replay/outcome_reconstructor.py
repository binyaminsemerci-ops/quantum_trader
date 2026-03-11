"""
OutcomeReconstructor — Reconstructs post-decision price path and outcomes.

Phase 5 replay component. Read-only.

Reads from: Redis position snapshots (historical), kline data
Writes to: Nothing (pure computation)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default evaluation horizon: 4 hours
DEFAULT_HORIZON_SEC = 14400.0

# Minimum price samples needed for a valid reconstruction
MIN_PRICE_SAMPLES = 10

# ASSUMPTION: Price path is reconstructed from Redis position snapshots
# that are written periodically by the existing position state builder.
# If kline data is available, it provides better granularity.
SNAPSHOT_KEY_PATTERN = "quantum:position:snapshot:{symbol}"


@dataclass(frozen=True)
class OutcomePathResult:
    """Reconstructed post-decision price and PnL path."""

    price_path: List[float] = field(default_factory=list)
    pnl_path: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    peak_pnl: float = 0.0
    peak_pnl_timestamp: float = 0.0
    trough_pnl: float = 0.0
    trough_pnl_timestamp: float = 0.0
    max_drawdown_after_peak: float = 0.0
    final_pnl: float = 0.0
    final_price: float = 0.0
    sample_count: int = 0
    quality_flags: Tuple[str, ...] = ()


class OutcomeReconstructor:
    """
    Reconstructs the post-decision price path for a position.

    Uses Redis position snapshots and/or kline data to build
    a time-series of prices and PnL after a decision point.

    Read-only. Never writes to any stream or key.
    """

    def __init__(self, redis_client) -> None:
        self._r = redis_client

    def reconstruct(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        decision_timestamp: float,
        horizon_seconds: float = DEFAULT_HORIZON_SEC,
        quantity: float = 1.0,
    ) -> OutcomePathResult:
        """
        Build post-decision price/PnL path for a position.

        Args:
            symbol: Trading pair.
            side: "LONG" or "SHORT".
            entry_price: Position entry price.
            decision_timestamp: Epoch of the decision.
            horizon_seconds: How far forward to look.
            quantity: Position size for PnL computation.

        Returns:
            OutcomePathResult with price/PnL path and summary stats.
        """
        end_ts = decision_timestamp + horizon_seconds
        quality_flags: List[str] = []

        # Attempt to load price samples from position snapshot stream
        prices, timestamps = self._load_price_samples(
            symbol, decision_timestamp, end_ts
        )

        if len(prices) < MIN_PRICE_SAMPLES:
            quality_flags.append("INSUFFICIENT_PRICE_SAMPLES")
            if not prices:
                quality_flags.append("MISSING_PRICE_PATH")
                return OutcomePathResult(quality_flags=tuple(quality_flags))

        # Compute PnL path
        pnl_path = self._compute_pnl_path(prices, entry_price, side, quantity)

        # Find peak and trough
        peak_pnl = max(pnl_path) if pnl_path else 0.0
        peak_idx = pnl_path.index(peak_pnl) if pnl_path else 0
        peak_ts = timestamps[peak_idx] if peak_idx < len(timestamps) else 0.0

        trough_pnl = min(pnl_path) if pnl_path else 0.0
        trough_idx = pnl_path.index(trough_pnl) if pnl_path else 0
        trough_ts = timestamps[trough_idx] if trough_idx < len(timestamps) else 0.0

        # Max drawdown after peak
        max_dd = self._compute_max_drawdown_after_peak(pnl_path)

        return OutcomePathResult(
            price_path=prices,
            pnl_path=pnl_path,
            timestamps=timestamps,
            peak_pnl=peak_pnl,
            peak_pnl_timestamp=peak_ts,
            trough_pnl=trough_pnl,
            trough_pnl_timestamp=trough_ts,
            max_drawdown_after_peak=max_dd,
            final_pnl=pnl_path[-1] if pnl_path else 0.0,
            final_price=prices[-1] if prices else 0.0,
            sample_count=len(prices),
            quality_flags=tuple(quality_flags),
        )

    def _load_price_samples(
        self,
        symbol: str,
        start_ts: float,
        end_ts: float,
    ) -> Tuple[List[float], List[float]]:
        """
        Load historical price samples from position snapshot stream.

        ASSUMPTION: quantum:stream:exit.state.shadow contains mark_price
        entries that can be used to reconstruct price path.
        Falls back to empty if unavailable.
        """
        prices: List[float] = []
        timestamps: List[float] = []

        try:
            stream = "quantum:stream:exit.state.shadow"
            start_id = f"{int(start_ts * 1000)}-0"
            end_id = f"{int(end_ts * 1000)}-18446744073709551615"
            raw = self._r.xrange(stream, min=start_id, max=end_id, count=5000)

            for _entry_id, fields in raw:
                sym = fields.get(b"symbol", b"").decode("utf-8")
                if sym != symbol:
                    continue
                price_raw = fields.get(b"mark_price") or fields.get(b"current_price")
                if price_raw is None:
                    continue
                try:
                    price = float(price_raw)
                    ts_raw = fields.get(b"ts") or fields.get(b"open_timestamp")
                    ts = float(ts_raw) if ts_raw else start_ts
                    prices.append(price)
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            logger.error("[OutcomeReconstructor] Failed to load prices for %s: %s", symbol, e)

        return prices, timestamps

    @staticmethod
    def _compute_pnl_path(
        prices: List[float],
        entry_price: float,
        side: str,
        quantity: float,
    ) -> List[float]:
        """Compute PnL at each price point."""
        if entry_price <= 0 or not prices:
            return []
        pnl_path: List[float] = []
        for p in prices:
            if side == "LONG":
                pnl = (p - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - p) * quantity
            pnl_path.append(pnl)
        return pnl_path

    @staticmethod
    def _compute_max_drawdown_after_peak(pnl_path: List[float]) -> float:
        """Find the maximum drawdown from peak in the PnL path."""
        if not pnl_path:
            return 0.0
        running_peak = pnl_path[0]
        max_dd = 0.0
        for pnl in pnl_path:
            if pnl > running_peak:
                running_peak = pnl
            dd = running_peak - pnl
            if dd > max_dd:
                max_dd = dd
        return max_dd

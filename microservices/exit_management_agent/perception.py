"""perception: compute position metrics needed for exit decisions.

PerceptionEngine is stateful: it tracks peak prices in-memory per symbol.
Peak prices reset on service restart — this is documented and expected.
age_sec is also a lower bound (time since first observation in this process).

Exposed pure helper functions (_compute_r_net, _compute_distance_to_sl,
_compute_giveback) are importable directly in tests without instantiating
the engine.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from .models import PerceptionResult, PositionSnapshot
from .redis_io import RedisClient

_log = logging.getLogger("exit_management_agent.perception")

# ── Optional import: common.risk_settings ─────────────────────────────────────
# Gracefully degrade if the module is not on sys.path (e.g. unit tests).
try:
    from common.risk_settings import DEFAULT_SETTINGS, compute_harvest_r_targets  # type: ignore[import]

    _RISK_SETTINGS_AVAILABLE = True
except ImportError:
    _RISK_SETTINGS_AVAILABLE = False
    DEFAULT_SETTINGS = None  # type: ignore[assignment]


class PerceptionEngine:
    """
    Stateful perception engine.

    Tracks peak prices per symbol in-memory.  Call forget(symbol) when a
    position is closed so stale peak data does not leak into new positions.
    """

    def __init__(self, redis: Optional[RedisClient] = None) -> None:
        self._redis = redis
        # In-memory best-price table: symbol -> float
        # LONG: highest price observed; SHORT: lowest price observed.
        self._peak_prices: dict = {}

    async def compute(
        self,
        snapshot: PositionSnapshot,
        age_sec: float,
    ) -> PerceptionResult:
        """
        Compute a PerceptionResult for one position snapshot.

        Args:
            snapshot: Parsed position data from Redis hash.
            age_sec:  Lower-bound age in seconds (managed by the caller).
        """
        # ── 1. mark_price: prefer live ticker; fall back to hash-derived value ──
        mark_price = snapshot.mark_price
        if self._redis is not None:
            try:
                ticker_price = await self._redis.get_mark_price_from_ticker(
                    snapshot.symbol
                )
                if ticker_price is not None and ticker_price > 0.0:
                    mark_price = ticker_price
            except Exception as exc:
                _log.debug("%s: ticker lookup failed: %s", snapshot.symbol, exc)

        # ── 2. Peak price tracking ─────────────────────────────────────────────
        sym = snapshot.symbol
        prev_peak = self._peak_prices.get(sym)
        if prev_peak is None:
            peak_price = mark_price
        elif snapshot.is_long:
            peak_price = max(prev_peak, mark_price)
        else:
            peak_price = min(prev_peak, mark_price)
        self._peak_prices[sym] = peak_price

        # ── 3. R_net ───────────────────────────────────────────────────────────
        r_net = _compute_r_net(snapshot, mark_price)

        # ── 4. Distance to SL ──────────────────────────────────────────────────
        dist_to_sl = _compute_distance_to_sl(snapshot, mark_price)

        # ── 5. Giveback ────────────────────────────────────────────────────────
        giveback = _compute_giveback(snapshot, mark_price, peak_price)

        # ── 6. Leverage-scaled R targets ───────────────────────────────────────
        r_t1, r_lock = _get_r_targets(snapshot.leverage)

        return PerceptionResult(
            snapshot=snapshot,
            R_net=r_net,
            peak_price=peak_price,
            age_sec=age_sec,
            distance_to_sl_pct=dist_to_sl,
            giveback_pct=giveback,
            r_effective_t1=r_t1,
            r_effective_lock=r_lock,
        )

    def forget(self, symbol: str) -> None:
        """Remove in-memory peak tracking for a symbol (call on position close)."""
        self._peak_prices.pop(symbol, None)


# ── Pure helper functions (also used by tests) ─────────────────────────────────


def _compute_r_net(snapshot: PositionSnapshot, mark_price: float) -> float:
    """
    Compute current R-multiple.

    Priority:
      1. entry_risk_usdt from hash (most accurate)
      2. Stop-loss distance × quantity (fallback)
      3. PnL as fraction of notional (last resort; very rough)

    Returns 0.0 if nothing computable.
    """
    if snapshot.entry_risk_usdt > 0.0:
        if snapshot.is_long:
            pnl = (mark_price - snapshot.entry_price) * snapshot.quantity
        else:
            pnl = (snapshot.entry_price - mark_price) * snapshot.quantity
        return pnl / snapshot.entry_risk_usdt

    if snapshot.stop_loss > 0.0 and snapshot.entry_price > 0.0:
        sl_dist = abs(snapshot.entry_price - snapshot.stop_loss) * snapshot.quantity
        if sl_dist > 0.0:
            if snapshot.is_long:
                pnl = (mark_price - snapshot.entry_price) * snapshot.quantity
            else:
                pnl = (snapshot.entry_price - mark_price) * snapshot.quantity
            return pnl / sl_dist

    # Last resort: pnl as fraction of notional (not an R-multiple, but a proxy)
    notional = snapshot.entry_price * snapshot.quantity
    if notional > 0.0 and snapshot.unrealized_pnl != 0.0:
        return snapshot.unrealized_pnl / notional

    return 0.0


def _compute_distance_to_sl(
    snapshot: PositionSnapshot, mark_price: float
) -> float:
    """
    Distance from mark price to stop-loss, as a fraction of mark_price.

    Positive  → SL has buffer remaining.
    Negative  → current price has moved through SL.
    0.0       → no SL set.
    """
    if snapshot.stop_loss <= 0.0 or mark_price <= 0.0:
        return 0.0
    if snapshot.is_long:
        return (mark_price - snapshot.stop_loss) / mark_price
    else:
        return (snapshot.stop_loss - mark_price) / mark_price


def _compute_giveback(
    snapshot: PositionSnapshot,
    mark_price: float,
    peak_price: float,
) -> float:
    """
    Fraction [0.0–1.0] of peak unrealised profit that has been given back.

    0.0 → at or above peak (no giveback).
    1.0 → fully given back (price back at entry).
    > 1.0 is clipped to 1.0; < 0.0 is clipped to 0.0.
    """
    if snapshot.entry_price <= 0.0 or snapshot.quantity <= 0.0:
        return 0.0

    if snapshot.is_long:
        peak_profit = (peak_price - snapshot.entry_price) * snapshot.quantity
        current_profit = (mark_price - snapshot.entry_price) * snapshot.quantity
    else:
        peak_profit = (snapshot.entry_price - peak_price) * snapshot.quantity
        current_profit = (snapshot.entry_price - mark_price) * snapshot.quantity

    if peak_profit <= 0.0:
        return 0.0

    giveback = (peak_profit - current_profit) / peak_profit
    return min(max(giveback, 0.0), 1.0)


def _get_r_targets(leverage: float) -> tuple:
    """
    Return (r_effective_t1, r_effective_lock) scaled for the given leverage.

    Mirrors the formula in common/risk_settings.py:
        R_effective = R_base / sqrt(leverage)

    Uses common.risk_settings if available; otherwise reproduces the formula
    with the same base values so tests pass without the full project on path.
    """
    if _RISK_SETTINGS_AVAILABLE:
        targets = compute_harvest_r_targets(leverage, DEFAULT_SETTINGS)
        return targets["T1_R"], targets["lock_R"]

    # Fallback: same formula, same defaults as DEFAULT_SETTINGS
    scale = math.sqrt(max(float(leverage), 1.0))
    return 2.0 / scale, 1.5 / scale

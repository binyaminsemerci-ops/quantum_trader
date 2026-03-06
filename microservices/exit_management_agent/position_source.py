"""position_source: parse open positions from quantum:position:* Redis hashes.

Hash field map (written by harvest_brain and harvest_brain startup sync):
    symbol          — not stored as field; extracted from key name
    side            — "LONG" | "SHORT"
    quantity        — float str   ← NOTE: field is "quantity", NOT "position_qty"
    entry_price     — float str
    unrealized_pnl  — float str   (may be 0.0 if harvest_brain hasn't run)
    leverage        — float str
    stop_loss       — float str   (optional)
    take_profit     — float str   (optional)
    entry_risk_usdt — float str   (optional; computed by _enrich_position_from_redis)
    sync_timestamp  — int str     (unix epoch; rolling last-sync time)
    risk_missing    — "0"|"1"     (informational; checked by harvest_brain)
    source          — string      (informational)

mark_price is NOT stored directly; it is derived from unrealized_pnl, or
fetched from quantum:ticker:{symbol}.markPrice by the caller (PerceptionEngine).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .models import PositionSnapshot
from .redis_io import RedisClient

_log = logging.getLogger("exit_management_agent.position_source")


class PositionSource:
    """
    Read open positions from quantum:position:{symbol} Redis hashes.

    Each call to get_open_positions() issues a fresh SCAN + HGETALL.
    No caching; positions change every few seconds.
    """

    def __init__(self, redis: RedisClient, max_positions: int = 50) -> None:
        self._redis = redis
        self._max_positions = max_positions

    async def get_open_positions(
        self,
        allowlist: Optional[frozenset] = None,
    ) -> list:
        """
        Return PositionSnapshot objects for all currently open positions.

        Args:
            allowlist: If non-empty frozenset, only symbols in the set are included.
                       Pass None or empty frozenset to include all symbols.

        Returns:
            List of PositionSnapshot (empty on Redis error; fail-open).
        """
        try:
            keys = await self._redis.scan_position_keys()
        except Exception as exc:
            _log.error("SCAN quantum:position:* failed: %s", exc)
            return []

        snapshots = []
        for key in keys[: self._max_positions]:
            symbol = key.removeprefix("quantum:position:")

            if allowlist and symbol.upper() not in allowlist:
                continue

            try:
                data = await self._redis.hgetall_position(key)
            except Exception as exc:
                _log.warning("HGETALL failed for %s: %s", key, exc)
                continue

            snap = _parse_hash(symbol, data)
            if snap is not None:
                snapshots.append(snap)

        _log.debug("Loaded %d open positions", len(snapshots))
        return snapshots


# ── Parsing helpers ────────────────────────────────────────────────────────────


def _parse_hash(symbol: str, data: dict) -> Optional[PositionSnapshot]:
    """
    Convert a raw Redis hash dict into a PositionSnapshot.
    Returns None (and logs a warning) if essential fields are absent/invalid.
    Defensive: always uses .get() with fallbacks; never raises.
    """
    if not data:
        return None

    # ── Essential: quantity ────────────────────────────────────────────────
    try:
        qty = float(data.get("quantity", "") or 0)
    except (ValueError, TypeError):
        _log.warning("%s: unparseable quantity=%r — skipping", symbol, data.get("quantity"))
        return None

    if qty <= 0.0:
        _log.debug("%s: quantity=%s — closed/zero, skipping", symbol, qty)
        return None

    # ── Essential: entry_price ─────────────────────────────────────────────
    try:
        entry_price = float(data.get("entry_price", "") or 0)
    except (ValueError, TypeError):
        _log.warning("%s: unparseable entry_price=%r — skipping", symbol, data.get("entry_price"))
        return None

    if entry_price <= 0.0:
        _log.debug("%s: entry_price=%s — skipping", symbol, entry_price)
        return None

    # ── Side (normalise to LONG/SHORT) ─────────────────────────────────────
    side_raw = (data.get("side") or "LONG").strip().upper()
    if side_raw in ("BUY",):
        side = "LONG"
    elif side_raw in ("SELL",):
        side = "SHORT"
    elif side_raw in ("LONG", "SHORT"):
        side = side_raw
    else:
        _log.warning("%s: unexpected side=%r — defaulting to LONG", symbol, side_raw)
        side = "LONG"

    # ── Optional numeric fields ────────────────────────────────────────────
    leverage = max(_safe_float(data.get("leverage"), 1.0), 1.0)
    stop_loss = _safe_float(data.get("stop_loss"), 0.0)
    take_profit = _safe_float(data.get("take_profit"), 0.0)
    unrealized_pnl = _safe_float(data.get("unrealized_pnl"), 0.0)
    entry_risk_usdt = _safe_float(data.get("entry_risk_usdt"), 0.0)
    sync_ts = _safe_float(data.get("sync_timestamp"), time.time())

    # ── Derive mark_price from unrealized_pnl ─────────────────────────────
    mark_price = _derive_mark_price(
        side=side,
        entry_price=entry_price,
        quantity=qty,
        unrealized_pnl=unrealized_pnl,
    )

    return PositionSnapshot(
        symbol=symbol,
        side=side,
        quantity=qty,
        entry_price=entry_price,
        mark_price=mark_price,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        unrealized_pnl=unrealized_pnl,
        entry_risk_usdt=entry_risk_usdt,
        sync_timestamp=sync_ts,
    )


def _safe_float(value: Optional[str], default: float = 0.0) -> float:
    """Parse a string to float; return default on any error."""
    if not value:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _derive_mark_price(
    side: str,
    entry_price: float,
    quantity: float,
    unrealized_pnl: float,
) -> float:
    """
    Estimate current mark price from unrealized PnL.

    LONG:  mark = entry + pnl / qty
    SHORT: mark = entry - pnl / qty

    Falls back to entry_price when computation is not possible.
    The derived value may be stale if unrealized_pnl is stale;
    PerceptionEngine will override it with the ticker price when available.
    """
    if quantity <= 0.0 or entry_price <= 0.0:
        return entry_price
    pnl_per_unit = unrealized_pnl / quantity
    if side.upper() in ("LONG", "BUY"):
        estimated = entry_price + pnl_per_unit
    else:
        estimated = entry_price - pnl_per_unit
    return estimated if estimated > 0.0 else entry_price

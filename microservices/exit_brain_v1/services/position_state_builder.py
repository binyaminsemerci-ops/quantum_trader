"""
PositionStateBuilder — Reads from Redis, builds PositionExitState.

This is the ONLY module in exit_brain_v1 that performs Redis reads.
It does NOT write to any execution stream.

Data sources:
  [FACT]       quantum:position:snapshot:<symbol>  — P3.3 exchange snapshots
  [FACT]       quantum:position:ledger:<symbol>    — P3.3 ledger metadata
  [FACT]       quantum:stream:meta.regime          — MetaRegimeService latest
  [ASSUMPTION] quantum:marketstate:<symbol>        — MarketState publisher (key format TBD)

Shadow write (via ShadowPublisher):
  quantum:stream:exit.state.shadow
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from ..models.position_exit_state import PositionExitState

logger = logging.getLogger(__name__)

# Staleness thresholds (seconds)
SNAPSHOT_STALE_SEC = 30
REGIME_STALE_SEC = 120
MARKETSTATE_STALE_SEC = 120


class PositionStateBuilder:
    """
    Assembles PositionExitState from multiple Redis sources.

    Usage:
        builder = PositionStateBuilder(redis_client)
        state = builder.build("BTCUSDT")
        if state is None:
            # fail-closed: required data missing
    """

    def __init__(self, redis_client) -> None:
        """
        Args:
            redis_client: A synchronous redis.Redis instance.
        """
        self._r = redis_client

    # ── Public API ───────────────────────────────────────────────────────

    def build(self, symbol: str) -> Optional[PositionExitState]:
        """
        Build a PositionExitState for the given symbol.

        Fail-closed: returns None if REQUIRED data is missing.
        OPTIONAL data gaps are recorded in data_quality_flags.

        Args:
            symbol: Trading pair, e.g. "BTCUSDT".

        Returns:
            PositionExitState or None.
        """
        now = time.time()
        flags: List[str] = []
        source_ts: Dict[str, float] = {}

        # ── 1. Exchange snapshot (REQUIRED) ──────────────────────────────
        snapshot = self._read_snapshot(symbol)
        if snapshot is None:
            logger.warning("[ExitBrain] %s: no P3.3 snapshot — skip (fail-closed)", symbol)
            return None

        snap_ts = float(snapshot.get("ts_epoch", 0))
        source_ts["p33_snapshot"] = snap_ts
        if now - snap_ts > SNAPSHOT_STALE_SEC:
            flags.append("STALE_SNAPSHOT")

        position_amt = float(snapshot.get("position_amt", 0))
        if position_amt == 0:
            logger.debug("[ExitBrain] %s: position_amt=0, no open position", symbol)
            return None

        side = snapshot.get("side", "NONE")
        if side not in ("LONG", "SHORT"):
            logger.warning("[ExitBrain] %s: side=%s — skip", symbol, side)
            return None

        entry_price = float(snapshot.get("entry_price", 0))
        mark_price = float(snapshot.get("mark_price", 0))
        current_price = mark_price if mark_price > 0 else entry_price
        unrealized_pnl = float(snapshot.get("unrealized_pnl", 0))
        leverage = float(snapshot.get("leverage", 1))
        quantity = abs(position_amt)
        notional = current_price * quantity

        if entry_price <= 0 or current_price <= 0:
            logger.warning("[ExitBrain] %s: invalid prices entry=%.4f current=%.4f",
                           symbol, entry_price, current_price)
            return None

        # PnL percentage
        if side == "LONG":
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100

        # mark_price quality check
        if mark_price <= 0:
            flags.append("MARK_PRICE_FALLBACK")

        # ── 2. Ledger metadata (OPTIONAL) ────────────────────────────────
        ledger = self._read_ledger(symbol)
        open_timestamp = 0.0
        if ledger:
            # Use ledger updated_at as proxy for open time if available
            # TODO: P3.3 doesn't store actual entry_timestamp — track separately
            open_timestamp = float(ledger.get("updated_at", 0))

        if open_timestamp <= 0:
            # Fallback: use snapshot timestamp minus a small offset
            open_timestamp = snap_ts
            flags.append("OPEN_TIMESTAMP_ESTIMATED")

        # ── 3. MarketState (OPTIONAL) ────────────────────────────────────
        ms = self._read_market_state(symbol)
        volatility_short: Optional[float] = None
        volatility_medium: Optional[float] = None
        trend_signal: Optional[float] = None
        regime_probs: Optional[Dict[str, float]] = None

        if ms:
            ms_ts = float(ms.get("ts_timestamp", now))
            source_ts["market_state"] = ms_ts
            if now - ms_ts > MARKETSTATE_STALE_SEC:
                flags.append("STALE_MARKETSTATE")

            volatility_short = _safe_float(ms.get("sigma"))
            volatility_medium = _safe_float(ms.get("sigma"))  # same source for now
            trend_signal = _safe_float(ms.get("mu"))
            ts_val = _safe_float(ms.get("ts"))

            # Regime probs
            rp_raw = ms.get("regime_probs")
            if isinstance(rp_raw, str):
                try:
                    regime_probs = json.loads(rp_raw)
                except (json.JSONDecodeError, TypeError):
                    flags.append("REGIME_PROBS_PARSE_ERROR")
            elif isinstance(rp_raw, dict):
                regime_probs = rp_raw
        else:
            flags.append("MISSING_MARKETSTATE")

        # ── 4. Meta-regime (OPTIONAL) ────────────────────────────────────
        regime_label = "UNKNOWN"
        regime_confidence = 0.0

        regime_data = self._read_meta_regime()
        if regime_data:
            regime_ts = _parse_timestamp(regime_data.get("timestamp", regime_data.get("ts", 0)))
            if regime_ts > 0:
                source_ts["meta_regime"] = regime_ts
                if now - regime_ts > REGIME_STALE_SEC:
                    flags.append("STALE_REGIME")

            regime_label = regime_data.get("regime", "UNKNOWN")
            regime_confidence = float(regime_data.get("confidence", 0))
        else:
            flags.append("MISSING_REGIME")

        # ── 5. ATR (OPTIONAL) ────────────────────────────────────────────
        atr = self._read_atr(symbol)
        if atr is None:
            flags.append("MISSING_ATR")

        # ── Assemble ─────────────────────────────────────────────────────
        state = PositionExitState(
            position_id=f"{symbol}_{side}",
            symbol=symbol,
            side=side,
            status="OPEN",
            entry_price=entry_price,
            current_price=current_price,
            mark_price=mark_price,
            quantity=quantity,
            notional=notional,
            leverage=max(1.0, leverage),
            open_timestamp=open_timestamp,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            volatility_short=volatility_short,
            volatility_medium=volatility_medium,
            atr=atr,
            trend_signal=trend_signal,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            source_timestamps=source_ts,
            data_quality_flags=flags,
            shadow_only=True,
        )

        # Validate
        errors = state.validate()
        if errors:
            logger.error("[ExitBrain] %s: validation failed: %s", symbol, errors)
            return None

        return state

    def build_all(self, symbols: List[str]) -> List[PositionExitState]:
        """Build states for all symbols. Skips failures silently."""
        results = []
        for sym in symbols:
            state = self.build(sym)
            if state is not None:
                results.append(state)
        return results

    def discover_open_positions(self) -> List[str]:
        """
        Discover symbols with open positions from P3.3 snapshots.

        Scans quantum:position:snapshot:* and returns symbols where
        position_amt != 0.
        """
        symbols = []
        cursor = 0
        while True:
            cursor, keys = self._r.scan(
                cursor=cursor,
                match="quantum:position:snapshot:*",
                count=100,
            )
            for key in keys:
                key_str = key if isinstance(key, str) else key.decode("utf-8")
                # Extract symbol from key
                symbol = key_str.split(":")[-1]
                # Quick check: non-zero position
                amt = self._r.hget(key_str, "position_amt")
                if amt is not None:
                    try:
                        if float(amt if isinstance(amt, str) else amt.decode("utf-8")) != 0:
                            symbols.append(symbol)
                    except (ValueError, AttributeError):
                        pass
            if cursor == 0:
                break
        return symbols

    # ── Private readers ──────────────────────────────────────────────────

    def _read_snapshot(self, symbol: str) -> Optional[Dict]:
        """Read P3.3 exchange snapshot."""
        key = f"quantum:position:snapshot:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return _decode_hash(data)

    def _read_ledger(self, symbol: str) -> Optional[Dict]:
        """Read P3.3 ledger metadata."""
        key = f"quantum:position:ledger:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return _decode_hash(data)

    def _read_market_state(self, symbol: str) -> Optional[Dict]:
        """
        Read MarketState from Redis.

        ASSUMPTION: Key format is quantum:marketstate:<symbol> (hash or string).
        If format differs on VPS, this method needs updating.
        """
        # Try hash first
        key = f"quantum:marketstate:{symbol}"
        data = self._r.hgetall(key)
        if data:
            return _decode_hash(data)

        # Try JSON string
        raw = self._r.get(key)
        if raw:
            try:
                return json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except (json.JSONDecodeError, TypeError):
                return None

        return None

    def _read_meta_regime(self) -> Optional[Dict]:
        """Read latest entry from quantum:stream:meta.regime."""
        try:
            entries = self._r.xrevrange("quantum:stream:meta.regime", count=1)
            if not entries:
                return None
            _id, data = entries[0]
            return _decode_hash(data)
        except Exception as e:
            logger.debug("[ExitBrain] meta.regime read error: %s", e)
            return None

    def _read_atr(self, symbol: str) -> Optional[float]:
        """
        Read ATR value from Redis.

        Tries multiple key patterns used in the system.
        """
        for key_pattern in [
            f"quantum:atr:{symbol}",
            f"quantum:indicator:atr:{symbol}",
            f"atr:{symbol}",
        ]:
            val = self._r.get(key_pattern)
            if val is not None:
                return _safe_float(val)
        return None


# ── Utilities ────────────────────────────────────────────────────────────

def _decode_hash(data: Dict) -> Dict:
    """Decode bytes keys/values from Redis HGETALL to str."""
    result = {}
    for k, v in data.items():
        key = k if isinstance(k, str) else k.decode("utf-8")
        val = v if isinstance(v, str) else v.decode("utf-8")
        result[key] = val
    return result


def _safe_float(val) -> Optional[float]:
    """Convert to float safely, return None on failure."""
    if val is None:
        return None
    try:
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_timestamp(val) -> float:
    """Parse epoch float or ISO-8601 string to epoch seconds. Returns 0.0 on failure."""
    if val is None or val == 0:
        return 0.0
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val)
    try:
        return float(val_str)
    except ValueError:
        pass
    # Try ISO-8601 parse
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(val_str, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return 0.0

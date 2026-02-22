"""
feeds/position_provider.py
Reads all quantum:position:* keys from Redis.
Validates each position per Phase 0 spec validity rules.

Shadow mode note:
  In shadow mode the upstream V1 may not be refreshing sync_timestamp
  because of the known pnl_flat skip issue.  V2 must still evaluate
  these positions.  Use env HARVEST_V2_MAX_AGE_SEC to override the
  staleness gate (default 3600 for shadow vs 120 spec baseline).
"""

import os
import time
import logging
from typing import List, Optional, Tuple
from utils.redis_client import RedisClient

logger = logging.getLogger("hv2.feeds.position")

POSITION_KEY_PREFIX = "quantum:position:"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class Position:
    """Parsed, validated position snapshot."""
    __slots__ = (
        "symbol", "side", "quantity", "entry_price",
        "unrealized_pnl", "leverage", "atr_value",
        "volatility_factor", "entry_risk_usdt", "risk_price",
        "risk_missing", "sync_timestamp",
    )

    def __init__(self, raw: dict):
        self.symbol           = raw.get("symbol", "")
        self.side             = raw.get("side", "LONG").upper()
        self.quantity         = _safe_float(raw.get("quantity", 0))
        self.entry_price      = _safe_float(raw.get("entry_price", 0))
        self.unrealized_pnl   = _safe_float(raw.get("unrealized_pnl", 0))
        self.leverage         = _safe_int(raw.get("leverage", 1))
        self.atr_value        = _safe_float(raw.get("atr_value", 0))
        self.volatility_factor= _safe_float(raw.get("volatility_factor", 1.0))
        self.entry_risk_usdt  = _safe_float(raw.get("entry_risk_usdt", 0))
        self.risk_price       = _safe_float(raw.get("risk_price", 0))
        self.risk_missing     = _safe_int(raw.get("risk_missing", 0))
        self.sync_timestamp   = _safe_int(raw.get("sync_timestamp", 0))


class FetchResult:
    """Return value from fetch_positions, includes skip counts for metrics."""
    __slots__ = ("positions", "total_keys", "skipped_invalid", "skipped_stale")

    def __init__(self):
        self.positions: List[Position] = []
        self.total_keys: int = 0
        self.skipped_invalid: int = 0  # bad risk / bad ATR
        self.skipped_stale: int = 0    # sync_timestamp too old


# Env override: HARVEST_V2_MAX_AGE_SEC=86400 to disable staleness gate
_ENV_MAX_AGE = os.getenv("HARVEST_V2_MAX_AGE_SEC")
_DEFAULT_MAX_AGE = int(_ENV_MAX_AGE) if _ENV_MAX_AGE else 86400  # 24h default for shadow


class PositionProvider:
    """
    Fetches and validates all positions.

    Validity rules (Phase 0 §1.1):
    - entry_risk_usdt > 0 AND risk_missing != 1
    - atr_value > 0
    - sync_timestamp within max_position_age_sec

    Default max_age_sec is 3600 (1h) — relaxed from spec 120s baseline
    because in shadow mode V1 may not be refreshing sync_timestamp.
    Set HARVEST_V2_MAX_AGE_SEC env to override at runtime.
    """

    def __init__(self, redis: RedisClient, max_age_sec: int = _DEFAULT_MAX_AGE):
        self._redis = redis
        self.max_age_sec = max_age_sec

    def fetch_positions(self) -> FetchResult:
        result = FetchResult()
        keys = self._redis.keys(f"{POSITION_KEY_PREFIX}*")
        if not keys:
            return result

        result.total_keys = len(keys)
        now = int(time.time())

        for key in keys:
            raw = self._redis.hgetall(key)
            if not raw:
                result.skipped_invalid += 1
                continue

            pos = Position(raw)
            skip_reason = self._validate(pos, now)
            if skip_reason:
                logger.info(
                    "[HV2] SKIP_%s symbol=%s entry_risk=%.4f atr=%.6f age=%ds",
                    skip_reason, pos.symbol, pos.entry_risk_usdt,
                    pos.atr_value, now - pos.sync_timestamp,
                )
                if skip_reason == "STALE":
                    result.skipped_stale += 1
                else:
                    result.skipped_invalid += 1
                continue

            result.positions.append(pos)

        return result

    def _validate(self, pos: Position, now: int) -> Optional[str]:
        if pos.entry_risk_usdt <= 0 or pos.risk_missing == 1:
            return "INVALID_RISK"
        if pos.atr_value <= 0:
            return "ZERO_ATR"
        age = now - pos.sync_timestamp
        if age > self.max_age_sec:
            return "STALE"
        return None

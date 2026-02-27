"""
engine/state.py
Per-symbol runtime state: ATR ring buffer, trailing max, partial stage,
emission guard tracking, and Redis persistence.

Redis key: quantum:harvest_v2:state:{SYMBOL}
"""

import time
import logging
from collections import deque
from typing import Optional, Dict
import numpy as np

from utils.redis_client import RedisClient
from feeds.position_provider import Position

logger = logging.getLogger("hv2.state")

STATE_KEY_PREFIX = "quantum:harvest_v2:state:"


def _safe_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class SymbolState:
    """
    Runtime state for one symbol.
    ATR ring buffer is in-memory only (not persisted).
    All other fields are persisted to Redis after each emission.
    """

    def __init__(self, symbol: str, atr_window: int):
        self.symbol: str = symbol
        self.atr_window: int = atr_window

        # In-memory ATR ring buffer — not persisted
        self._atr_buf: deque = deque(maxlen=atr_window)

        # Persisted state fields
        self.max_R_seen: Optional[float] = None
        self.partial_stage: int = 0          # 0/1/2/3
        self.last_decision: Optional[str] = None
        self.last_emit_R: Optional[float] = None
        self.last_update_ts: float = 0.0

    # ------------------------------------------------------------------ #
    #  ATR / Regime                                                       #
    # ------------------------------------------------------------------ #

    def push_atr(self, atr_value: float):
        self._atr_buf.append(atr_value)

    def detect_regime(self) -> str:
        """
        Returns LOW_VOL / MID_VOL / HIGH_VOL based on ATR percentile.
        Defaults to MID_VOL on cold start (buffer < 2 samples).
        """
        if len(self._atr_buf) < 2:
            return "MID_VOL"

        arr = np.array(self._atr_buf)
        current = arr[-1]
        pct = float(np.sum(arr <= current) / len(arr) * 100.0)

        if pct < 20.0:
            return "LOW_VOL"
        elif pct < 80.0:
            return "MID_VOL"
        else:
            return "HIGH_VOL"

    # ------------------------------------------------------------------ #
    #  Trailing max                                                       #
    # ------------------------------------------------------------------ #

    def update_max_R(self, R_net: float):
        if self.max_R_seen is None or R_net > self.max_R_seen:
            self.max_R_seen = R_net

    def trailing_triggered(self, R_net: float, trailing_step: float) -> bool:
        """True when R_net has pulled back from peak by >= trailing_step."""
        if self.max_R_seen is None:
            return False
        return R_net < (self.max_R_seen - trailing_step)

    # ------------------------------------------------------------------ #
    #  Emission guard                                                     #
    # ------------------------------------------------------------------ #

    def should_emit(self, decision: str, R_net: float, r_emit_step: float) -> bool:
        """
        Emit if:
        A: decision changed from last emission
        B: abs(R_net - last_emit_R) > r_emit_step
        """
        if decision == "HOLD_SUPPRESSED":
            return False
        cond_a = (decision != self.last_decision)
        cond_b = (
            self.last_emit_R is None or
            abs(R_net - self.last_emit_R) > r_emit_step
        )
        return cond_a or cond_b

    def record_emission(self, decision: str, R_net: float):
        self.last_decision = decision
        self.last_emit_R   = R_net
        self.last_update_ts = time.time()


class StateManager:
    """
    Manages per-symbol SymbolState objects.
    Loads persisted state from Redis on first access.
    Writes state to Redis after each emission.
    """

    def __init__(self, redis: RedisClient, atr_window: int = 50):
        self._redis = redis
        self._atr_window = atr_window
        self._cache: Dict[str, SymbolState] = {}

    def set_atr_window(self, atr_window: int):
        self._atr_window = atr_window

    def get(self, pos: Position) -> SymbolState:
        """
        Returns SymbolState for symbol, loading from Redis if not yet cached.
        Always pushes current ATR into ring buffer.
        """
        symbol = pos.symbol
        if symbol not in self._cache:
            state = SymbolState(symbol, self._atr_window)
            self._load_from_redis(state)
            self._cache[symbol] = state

        state = self._cache[symbol]
        state.push_atr(pos.atr_value)
        return state

    def save(self, state: SymbolState):
        """Persist mutable state fields to Redis."""
        key = f"{STATE_KEY_PREFIX}{state.symbol}"
        mapping = {
            "max_R_seen":     "" if state.max_R_seen is None else f"{state.max_R_seen:.6f}",
            "partial_stage":  str(state.partial_stage),
            "last_decision":  state.last_decision or "",
            "last_emit_R":    "" if state.last_emit_R is None else f"{state.last_emit_R:.6f}",
            "last_update_ts": f"{state.last_update_ts:.3f}",
        }
        self._redis.hset(key, mapping)

    def _load_from_redis(self, state: SymbolState):
        key = f"{STATE_KEY_PREFIX}{state.symbol}"
        raw = self._redis.hgetall(key)
        if not raw:
            return  # fresh state with defaults

        state.max_R_seen    = _safe_float(raw.get("max_R_seen", ""), None) \
                              if raw.get("max_R_seen") else None
        state.partial_stage = _safe_int(raw.get("partial_stage", "0"), 0)
        state.last_decision = raw.get("last_decision") or None
        state.last_emit_R   = _safe_float(raw.get("last_emit_R", ""), None) \
                              if raw.get("last_emit_R") else None
        state.last_update_ts = _safe_float(raw.get("last_update_ts", "0"), 0.0)

        logger.debug(
            "[HV2] State loaded %s: partial=%d last=%s maxR=%s",
            state.symbol, state.partial_stage,
            state.last_decision, state.max_R_seen,
        )

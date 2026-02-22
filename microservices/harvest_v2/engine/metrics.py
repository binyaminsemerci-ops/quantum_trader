"""
engine/metrics.py
Atomic metric writes to quantum:metrics:harvest_v2.
Per Phase 0 spec §7.
"""

import time
import logging
from utils.redis_client import RedisClient

logger = logging.getLogger("hv2.metrics")


class MetricsWriter:
    def __init__(self, redis: RedisClient, key: str):
        self._redis = redis
        self._key = key

    def record_start(self):
        self._redis.hset(self._key, {"start_ts": f"{time.time():.3f}"})

    def tick(self, scanned: int, evaluated: int, emitted: int,
             skipped_invalid: int, hold_suppressed: int):
        r = self._redis.raw.pipeline()
        r.hincrby(self._key, "ticks", 1)
        r.hincrby(self._key, "evaluated", evaluated)
        r.hincrby(self._key, "skipped_invalid", skipped_invalid)
        r.hincrby(self._key, "hold_suppressed", hold_suppressed)
        r.hset(self._key, "last_tick_ts", f"{time.time():.3f}")
        r.execute()

    def emission(self, decision: str, regime: str, R_net: float):
        r = self._redis.raw.pipeline()
        if "FULL_CLOSE" in decision:
            r.hincrby(self._key, "full_closes", 1)
        elif "PARTIAL" in decision:
            r.hincrby(self._key, "partials", 1)
        elif decision == "HOLD":
            r.hincrby(self._key, "holds", 1)

        regime_field = {
            "LOW_VOL":  "regime_low",
            "MID_VOL":  "regime_mid",
            "HIGH_VOL": "regime_high",
        }.get(regime, "regime_mid")
        r.hincrby(self._key, regime_field, 1)
        r.hincrbyfloat(self._key, "avg_R_sum", R_net)
        r.hincrby(self._key, "avg_R_count", 1)
        r.execute()

    def divergence(self):
        self._redis.hincrby(self._key, "divergence_from_v1", 1)

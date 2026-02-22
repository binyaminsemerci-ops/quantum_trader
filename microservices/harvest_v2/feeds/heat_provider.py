"""
feeds/heat_provider.py
Reads heat from quantum:capital:state.
SAFETY RULE: always returns 0.0 if key or field is missing.
"""

import logging
from utils.redis_client import RedisClient

logger = logging.getLogger("hv2.feeds.heat")

HEAT_KEY   = "quantum:capital:state"
HEAT_FIELD = "heat"


class HeatProvider:
    def __init__(self, redis: RedisClient):
        self._redis = redis

    def get_heat(self) -> float:
        """
        Returns heat in [0.0, 1.0].
        Defaults to 0.0 on any missing/invalid value — non-negotiable.
        """
        try:
            raw = self._redis.hget(HEAT_KEY, HEAT_FIELD)
            if raw is None:
                return 0.0
            heat = float(raw)
            # Clamp unconditionally — no exception can raise heat above 1.0
            return max(0.0, min(1.0, heat))
        except (TypeError, ValueError):
            logger.warning("[HV2] heat parse error — defaulting to 0.0")
            return 0.0
        except Exception as exc:
            logger.error("[HV2] heat read exception: %s — defaulting to 0.0", exc)
            return 0.0

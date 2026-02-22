"""
engine/config.py
Loads harvest_v2 config from Redis hash quantum:config:harvest_v2.
All fields have safe defaults per Phase 0 spec.
"""

import time
import logging
from dataclasses import dataclass, fields
from utils.redis_client import RedisClient

logger = logging.getLogger("hv2.config")

CONFIG_REDIS_KEY = "quantum:config:harvest_v2"


@dataclass
class HarvestV2Config:
    # R thresholds
    r_stop_base: float             = 0.5
    r_target_base: float           = 3.0
    trailing_step: float           = 0.3
    r_emit_step: float             = 0.05

    # Partial scaling
    partial_25_r: float            = 1.0
    partial_50_r: float            = 1.5
    partial_75_r: float            = 2.0

    # Vol factors per regime
    vol_factor_low: float          = 0.7
    vol_factor_mid: float          = 1.0
    vol_factor_high: float         = 1.4

    # Heat
    heat_sensitivity: float        = 0.5

    # ATR window
    atr_window: int                = 50

    # Timing
    config_refresh_interval_sec: int = 60
    max_position_age_sec: int        = 86400  # 24h — shadow mode: V1 doesn't refresh sync_timestamp
    scan_interval_sec: float         = 2.0

    # Streams / keys
    stream_shadow: str             = "quantum:stream:harvest.v2.shadow"
    stream_live: str               = ""   # empty = shadow-only; set to quantum:stream:apply.plan to go live
    metrics_key: str               = "quantum:metrics:harvest_v2"


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


class ConfigLoader:
    """
    Loads config from Redis on startup and re-loads every
    config_refresh_interval_sec seconds.
    """

    def __init__(self, redis: RedisClient):
        self._redis = redis
        self._config: HarvestV2Config = HarvestV2Config()
        self._last_load: float = 0.0
        self._load()

    def get(self) -> HarvestV2Config:
        now = time.monotonic()
        if now - self._last_load >= self._config.config_refresh_interval_sec:
            self._load()
        return self._config

    def _load(self):
        raw = self._redis.hgetall(CONFIG_REDIS_KEY)
        if not raw:
            logger.warning("[HV2] CONFIG_MISSING_USING_DEFAULTS — %s not found", CONFIG_REDIS_KEY)
            self._config = HarvestV2Config()
            self._last_load = time.monotonic()
            return

        cfg = HarvestV2Config()
        float_fields = {
            "r_stop_base", "r_target_base", "trailing_step", "r_emit_step",
            "partial_25_r", "partial_50_r", "partial_75_r",
            "vol_factor_low", "vol_factor_mid", "vol_factor_high",
            "heat_sensitivity", "scan_interval_sec",
        }
        int_fields = {
            "atr_window", "config_refresh_interval_sec",
            "max_position_age_sec",
        }
        str_fields = {"stream_shadow", "stream_live", "metrics_key"}

        field_names = {f.name for f in fields(cfg)}
        for key, val in raw.items():
            if key not in field_names:
                continue
            if key in float_fields:
                setattr(cfg, key, _safe_float(val, getattr(cfg, key)))
            elif key in int_fields:
                setattr(cfg, key, _safe_int(val, getattr(cfg, key)))
            elif key in str_fields:
                setattr(cfg, key, str(val))

        self._config = cfg
        self._last_load = time.monotonic()
        logger.debug("[HV2] Config loaded from Redis: %s", cfg)

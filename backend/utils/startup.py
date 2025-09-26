"""Startup helpers for the FastAPI application.

Provides a lightweight function to log configuration and available exchange
adapters during application startup. Kept intentionally small and side-effect
free so CI and tests that import it don't need external services.
"""

from __future__ import annotations

from typing import List
import logging
import json
import os

from config.config import load_config
from backend.utils.exchanges import _ADAPTER_REGISTRY
from typing import Dict


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("quantum_trader.startup")
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()

        # Simple JSON formatter
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                return json.dumps(payload)

        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger


async def log_startup_info() -> None:
    """Async-safe startup logger called from FastAPI startup event.

    Emits a JSON-friendly summary that includes available adapters,
    which credential sets are present (masked), and a small capability
    summary (which adapters implement spot/futures/create_order).
    """
    logger = _configure_logger()
    cfg = load_config()
    adapters: List[str] = list(_ADAPTER_REGISTRY.keys())

    # Masked presence of keys (don't print the actual secrets)
    key_presence: Dict[str, bool] = {
        "binance": bool(cfg.binance_api_key and cfg.binance_api_secret),
        "coinbase": bool(cfg.coinbase_api_key and cfg.coinbase_api_secret),
        "kucoin": bool(cfg.kucoin_api_key and cfg.kucoin_api_secret),
        "cryptopanic": bool(cfg.cryptopanic_key),
    }

    # Capability summary: probe adapter classes for expected methods
    caps: Dict[str, Dict[str, bool]] = {}
    for name, cls in _ADAPTER_REGISTRY.items():
        caps[name] = {
            "spot_balance": hasattr(cls, "spot_balance"),
            "futures_balance": hasattr(cls, "futures_balance"),
            "fetch_recent_trades": hasattr(cls, "fetch_recent_trades"),
            "create_order": hasattr(cls, "create_order"),
        }

    logger.info(
        "startup",
        extra={
            "adapters": adapters,
            "key_presence": key_presence,
            "capabilities": caps,
        },
    )

"""
SPRINT 1 - D5: TradeStore Migration
===================================

Unified trade persistence layer with Redis/SQLite backends.
"""

from backend.core.trading.trade_store_base import (
    Trade,
    TradeStatus,
    TradeSide,
    TradeStore,
    get_trade_store,
    reset_trade_store,
)
from backend.core.trading.trade_store_sqlite import TradeStoreSQLite
from backend.core.trading.trade_store_redis import TradeStoreRedis

__all__ = [
    "Trade",
    "TradeStatus",
    "TradeSide",
    "TradeStore",
    "TradeStoreSQLite",
    "TradeStoreRedis",
    "get_trade_store",
    "reset_trade_store",
]

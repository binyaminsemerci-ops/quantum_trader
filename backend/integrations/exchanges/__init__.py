"""
Multi-Exchange Abstraction Layer

EPIC-EXCH-003: Provides unified interface for trading across multiple exchanges.

Components:
- IExchangeClient: Protocol defining exchange interface
- Exchange-specific adapters (Binance, Bybit, OKX, KuCoin, Kraken, Firi)
- Exchange factory for creating clients
- Symbol-to-exchange routing
"""

from backend.integrations.exchanges.base import IExchangeClient, ExchangeAPIError
from backend.integrations.exchanges.models import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderRequest,
    OrderResult,
    CancelResult,
    Position,
    Balance,
    Kline,
    OrderStatus,
    PositionSide,
)
from backend.integrations.exchanges.factory import (
    ExchangeType,
    ExchangeConfig,
    get_exchange_client,
    resolve_exchange_for_symbol,
    set_symbol_exchange_mapping,
    load_symbol_mapping_from_policy,
    get_current_symbol_mapping,
)
from backend.integrations.exchanges.binance_adapter import BinanceAdapter
from backend.integrations.exchanges.bybit_adapter import BybitAdapter
from backend.integrations.exchanges.okx_adapter import OKXAdapter
from backend.integrations.exchanges.kucoin_adapter import KuCoinAdapter
from backend.integrations.exchanges.kraken_adapter import KrakenAdapter
from backend.integrations.exchanges.firi_adapter import FiriAdapter

__all__ = [
    # Protocol & exceptions
    "IExchangeClient",
    "ExchangeAPIError",
    # Enums
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "PositionSide",
    # Models
    "OrderRequest",
    "OrderResult",
    "CancelResult",
    "Position",
    "Balance",
    "Kline",
    # Factory
    "ExchangeType",
    "ExchangeConfig",
    "get_exchange_client",
    "resolve_exchange_for_symbol",
    "set_symbol_exchange_mapping",
    "load_symbol_mapping_from_policy",
    "get_current_symbol_mapping",
    # Adapters
    "BinanceAdapter",
    "BybitAdapter",
    "OKXAdapter",
    "KuCoinAdapter",
    "KrakenAdapter",
    "FiriAdapter",
]

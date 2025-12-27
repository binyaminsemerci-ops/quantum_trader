"""
Trade Replay Engine (TRE) - Time Machine for Quantum Trader

Replays historical market data through the full trading system pipeline,
reconstructing signals, decisions, trades, and system states for post-mortem analysis.
"""

from .replay_config import ReplayConfig, ReplayMode
from .replay_result import ReplayResult, TradeRecord, EventRecord, SymbolStats, StrategyStats
from .replay_context import ReplayContext, Position
from .replay_market_data import ReplayMarketDataSource, MarketDataClient
from .exchange_simulator import ExchangeSimulator, ExecutionResult
from .trade_replay_engine import TradeReplayEngine

__all__ = [
    "ReplayConfig",
    "ReplayMode",
    "ReplayResult",
    "TradeRecord",
    "EventRecord",
    "SymbolStats",
    "StrategyStats",
    "ReplayContext",
    "Position",
    "ReplayMarketDataSource",
    "MarketDataClient",
    "ExchangeSimulator",
    "ExecutionResult",
    "TradeReplayEngine",
]

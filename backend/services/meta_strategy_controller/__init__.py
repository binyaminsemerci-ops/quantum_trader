"""
Meta Strategy Controller for Quantum Trader.

The MSC AI is the top-level "brain" that analyzes market conditions
and sets global trading policy.
"""

from .controller import MetaStrategyController
try:
    from .controller import (
        MetricsRepository,
        StrategyRepository,
        StrategyStats,
        StrategyConfig,
        SystemHealth,
        RegimeType
    )
except ImportError:
    from typing import Protocol as TypingProtocol
    from datetime import datetime
    from dataclasses import dataclass
    from enum import Enum
    
    class MetricsRepository(TypingProtocol):
        """Interface for accessing system-wide metrics"""
        def get_current_drawdown_pct(self) -> float: ...
        def get_global_winrate(self, last_trades: int = 200) -> float: ...
        def get_equity_curve(self, days: int = 30) -> list[tuple[datetime, float]]: ...
        def get_global_regime(self) -> str: ...
        def get_volatility_level(self) -> str: ...
        def get_consecutive_losses(self) -> int: ...
        def get_days_since_last_profit(self) -> int: ...
    
    class StrategyRepository(TypingProtocol):
        """Interface for strategy repository"""
        def get_all_strategies(self) -> list: ...
        def get_strategy(self, strategy_id: str): ...
    
    # Fallback definitions for missing classes
    class RegimeType(Enum):
        TRENDING = "trending"
        RANGING = "ranging"
        VOLATILE = "volatile"
        CALM = "calm"
    
    @dataclass
    class StrategyStats:
        strategy_id: str
        total_trades: int = 0
        winrate: float = 0.0
        avg_r: float = 0.0
        total_pnl: float = 0.0
    
    @dataclass
    class StrategyConfig:
        strategy_id: str
        enabled: bool = True
        min_confidence: float = 0.65
    
    @dataclass
    class SystemHealth:
        drawdown_pct: float = 0.0
        equity_curve: list = None
        consecutive_losses: int = 0

from .models import MarketAnalysis, MarketRegime

__all__ = [
    "MetaStrategyController",
    "MetricsRepository",
    "StrategyRepository",
    "MarketAnalysis",
    "MarketRegime",
    "StrategyStats",
    "StrategyConfig",
    "SystemHealth",
    "RegimeType",
]

"""
Repository protocols for Strategy Generator AI.
"""

from typing import Protocol
from datetime import datetime

from .models import StrategyConfig, StrategyStats, StrategyStatus


class StrategyRepository(Protocol):
    """
    Persistence interface for strategies and their performance stats.
    
    Implementations can use SQL, NoSQL, or in-memory storage.
    """
    
    def save_strategy(self, config: StrategyConfig) -> None:
        """
        Persist a strategy configuration.
        
        Args:
            config: Strategy to save
        """
        ...
    
    def get_strategy(self, strategy_id: str) -> StrategyConfig | None:
        """
        Retrieve a strategy by ID.
        
        Args:
            strategy_id: Unique strategy identifier
            
        Returns:
            Strategy config or None if not found
        """
        ...
    
    def get_strategies_by_status(self, status: StrategyStatus) -> list[StrategyConfig]:
        """
        Get all strategies with given status.
        
        Args:
            status: Filter by strategy status
            
        Returns:
            List of matching strategies
        """
        ...
    
    def update_status(self, strategy_id: str, new_status: StrategyStatus) -> None:
        """
        Change strategy status (CANDIDATE -> SHADOW -> LIVE -> DISABLED).
        
        Args:
            strategy_id: Strategy to update
            new_status: New status value
        """
        ...
    
    def save_stats(self, stats: StrategyStats) -> None:
        """
        Save performance statistics for a strategy.
        
        Args:
            stats: Performance metrics to persist
        """
        ...
    
    def get_stats(
        self,
        strategy_id: str,
        source: str,
        days: int | None = None
    ) -> list[StrategyStats]:
        """
        Get stats for a strategy, optionally filtered by recent days.
        
        Args:
            strategy_id: Strategy to query
            source: "BACKTEST", "SHADOW", or "LIVE"
            days: If provided, only return stats from last N days
            
        Returns:
            List of stats ordered by date (newest first)
        """
        ...


class MarketDataClient(Protocol):
    """
    Interface for historical and live market data.
    """
    
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> "pd.DataFrame":
        """
        Returns OHLCV data.
        
        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            timeframe: Candle interval (e.g. "15m", "1h")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...

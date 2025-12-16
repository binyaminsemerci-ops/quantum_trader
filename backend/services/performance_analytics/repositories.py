"""
Performance & Analytics Layer (PAL) - Repository Protocols

Defines the interfaces for data access used by the analytics service.
"""

from datetime import datetime
from typing import Protocol, Optional

from .models import Trade, StrategyStats, SymbolStats, EventLog, EquityPoint


class TradeRepository(Protocol):
    """
    Repository for accessing trade history.
    """
    
    def get_trades(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbol: str | None = None,
        strategy_id: str | None = None,
        limit: int | None = None,
    ) -> list[Trade]:
        """
        Fetch trades with optional filters.
        
        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            symbol: Filter by symbol
            strategy_id: Filter by strategy
            limit: Max number of trades
        
        Returns:
            List of trades matching filters
        """
        ...
    
    def get_trade_count(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Get total trade count in period"""
        ...


class StrategyStatsRepository(Protocol):
    """
    Repository for strategy performance statistics.
    """
    
    def get_strategy_stats(
        self,
        strategy_id: str,
        days: int = 90
    ) -> list[StrategyStats]:
        """
        Get time-series stats for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            days: Number of days to look back
        
        Returns:
            List of stats points over time
        """
        ...
    
    def get_all_strategy_ids(self) -> list[str]:
        """Get list of all known strategy IDs"""
        ...


class SymbolStatsRepository(Protocol):
    """
    Repository for symbol performance statistics.
    """
    
    def get_symbol_stats(
        self,
        symbol: str,
        days: int = 90
    ) -> list[SymbolStats]:
        """
        Get time-series stats for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
        
        Returns:
            List of stats points over time
        """
        ...
    
    def get_all_symbols(self) -> list[str]:
        """Get list of all traded symbols"""
        ...


class MetricsRepository(Protocol):
    """
    Repository for global metrics (equity, drawdown, etc).
    """
    
    def get_equity_curve(
        self,
        days: int = 365
    ) -> list[EquityPoint]:
        """
        Get equity curve points.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of equity points over time
        """
        ...
    
    def get_drawdown_curve(
        self,
        days: int = 365
    ) -> list[tuple[datetime, float]]:
        """
        Get drawdown curve (timestamp, drawdown_pct).
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of (timestamp, drawdown) tuples
        """
        ...
    
    def get_current_equity(self) -> float:
        """Get current equity value"""
        ...
    
    def get_initial_balance(self) -> float:
        """Get initial balance"""
        ...


class EventLogRepository(Protocol):
    """
    Repository for system events.
    """
    
    def get_emergency_events(
        self,
        days: int = 365
    ) -> list[EventLog]:
        """
        Get emergency stop events.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of emergency stop events
        """
        ...
    
    def get_health_events(
        self,
        days: int = 365
    ) -> list[EventLog]:
        """
        Get health-related events.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of health events
        """
        ...
    
    def get_all_events(
        self,
        days: int = 365,
        event_types: list[str] | None = None,
    ) -> list[EventLog]:
        """
        Get all events with optional type filter.
        
        Args:
            days: Number of days to look back
            event_types: Filter by event types
        
        Returns:
            List of events
        """
        ...

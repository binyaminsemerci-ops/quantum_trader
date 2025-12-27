"""
Repository for storing and retrieving metrics.
"""

from typing import Dict, List, Optional, Protocol
from datetime import datetime, timedelta
import asyncio

from .models import (
    StrategyMetrics,
    SystemMetrics,
    ModelMetrics,
    TradeMetrics,
)


class MetricsRepository(Protocol):
    """Protocol for metrics storage."""
    
    async def save_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """Save strategy metrics."""
        ...
    
    async def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get latest strategy metrics."""
        ...
    
    async def save_model_metrics(self, metrics: ModelMetrics) -> None:
        """Save model metrics."""
        ...
    
    async def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get latest model metrics."""
        ...
    
    async def save_system_metrics(self, metrics: SystemMetrics) -> None:
        """Save system metrics."""
        ...
    
    async def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        ...
    
    async def save_trade_metrics(self, metrics: TradeMetrics) -> None:
        """Save trade metrics."""
        pass
    
    async def get_trade_history(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TradeMetrics]:
        """Get trade history."""
        ...


class InMemoryMetricsRepository:
    """In-memory implementation of MetricsRepository."""
    
    def __init__(self):
        self._strategy_metrics: Dict[str, StrategyMetrics] = {}
        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._system_metrics: Optional[SystemMetrics] = None
        self._trade_history: List[TradeMetrics] = []
        self._lock = asyncio.Lock()
    
    async def save_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """Save strategy metrics."""
        async with self._lock:
            self._strategy_metrics[metrics.strategy_id] = metrics
    
    async def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get latest strategy metrics."""
        return self._strategy_metrics.get(strategy_id)
    
    async def get_all_strategy_metrics(self) -> List[StrategyMetrics]:
        """Get all strategy metrics."""
        return list(self._strategy_metrics.values())
    
    async def save_model_metrics(self, metrics: ModelMetrics) -> None:
        """Save model metrics."""
        async with self._lock:
            key = f"{metrics.model_name}:{metrics.version}"
            self._model_metrics[key] = metrics
    
    async def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get latest model metrics."""
        # Find the latest version
        matching = [m for m in self._model_metrics.values() if m.model_name == model_name]
        if not matching:
            return None
        return max(matching, key=lambda m: m.timestamp)
    
    async def get_all_model_metrics(self) -> List[ModelMetrics]:
        """Get all model metrics."""
        return list(self._model_metrics.values())
    
    async def save_system_metrics(self, metrics: SystemMetrics) -> None:
        """Save system metrics."""
        async with self._lock:
            self._system_metrics = metrics
    
    async def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        return self._system_metrics
    
    async def save_trade_metrics(self, metrics: TradeMetrics) -> None:
        """Save trade metrics."""
        async with self._lock:
            self._trade_history.append(metrics)
            # Keep only last 10000 trades
            if len(self._trade_history) > 10000:
                self._trade_history = self._trade_history[-10000:]
    
    async def get_trade_history(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TradeMetrics]:
        """Get trade history."""
        trades = self._trade_history
        
        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        
        # Return most recent first
        return list(reversed(trades[-limit:]))
    
    async def get_trades_in_timerange(
        self,
        start: datetime,
        end: datetime,
        strategy_id: Optional[str] = None,
    ) -> List[TradeMetrics]:
        """Get trades within a time range."""
        trades = self._trade_history
        
        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        
        trades = [
            t for t in trades
            if start <= t.opened_at <= end
        ]
        
        return trades

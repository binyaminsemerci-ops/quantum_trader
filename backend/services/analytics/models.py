"""
Data models for Analytics & Reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class MetricType(Enum):
    """Types of metrics tracked."""
    STRATEGY = "strategy"
    SYSTEM = "system"
    MODEL = "model"
    TRADE = "trade"


@dataclass
class StrategyMetrics:
    """Performance metrics for a trading strategy."""
    strategy_id: str
    
    # Performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Returns
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Lifecycle
    stage: str = "SHADOW"  # SHADOW, LIVE, RETIRED
    promoted_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.losing_trades == 0 or self.avg_loss == 0:
            return float('inf') if self.winning_trades > 0 else 0.0
        
        gross_profit = self.winning_trades * abs(self.avg_win)
        gross_loss = self.losing_trades * abs(self.avg_loss)
        
        if gross_loss == 0:
            return float('inf')
        
        return gross_profit / gross_loss
    
    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "stage": self.stage,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelMetrics:
    """Performance metrics for ML models."""
    model_name: str
    version: str
    
    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Prediction quality
    predictions_made: int = 0
    correct_predictions: int = 0
    
    # Performance
    avg_inference_time_ms: float = 0.0
    
    # Lifecycle
    stage: str = "SHADOW"  # SHADOW, LIVE, RETIRED
    promoted_at: Optional[datetime] = None
    trained_at: Optional[datetime] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "stage": self.stage,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SystemMetrics:
    """Overall system performance metrics."""
    
    # Policy
    current_risk_mode: str = "NORMAL"
    policy_changes_count: int = 0
    last_policy_change: Optional[datetime] = None
    
    # Trading
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    
    # Performance
    total_pnl: float = 0.0
    total_volume: float = 0.0
    
    # Health
    health_status: str = "HEALTHY"
    uptime_seconds: float = 0.0
    
    # Events
    events_published: int = 0
    events_processed: int = 0
    event_errors: int = 0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "current_risk_mode": self.current_risk_mode,
            "policy_changes_count": self.policy_changes_count,
            "last_policy_change": self.last_policy_change.isoformat() if self.last_policy_change else None,
            "total_positions": self.total_positions,
            "open_positions": self.open_positions,
            "closed_positions": self.closed_positions,
            "total_pnl": self.total_pnl,
            "total_volume": self.total_volume,
            "health_status": self.health_status,
            "uptime_seconds": self.uptime_seconds,
            "events_published": self.events_published,
            "events_processed": self.events_processed,
            "event_errors": self.event_errors,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeMetrics:
    """Metrics for individual trades."""
    trade_id: str
    symbol: str
    strategy_id: str
    
    # Trade details
    side: str  # LONG/SHORT
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    
    # Performance
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    # Timing
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    def calculate_pnl(self) -> float:
        """Calculate PnL."""
        if self.exit_price is None:
            return 0.0
        
        if self.side == "LONG":
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.quantity
    
    def calculate_pnl_percent(self) -> float:
        """Calculate PnL percentage."""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == "LONG":
            return ((self.exit_price or self.entry_price) - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - (self.exit_price or self.entry_price)) / self.entry_price * 100
    
    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "duration_seconds": self.duration_seconds,
        }

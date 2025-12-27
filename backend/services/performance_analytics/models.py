"""
Performance & Analytics Layer (PAL) - Core Data Models

Defines the domain models used throughout the analytics system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeExitReason(Enum):
    """Trade exit reason"""
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "BULL"
    BEAR = "BEAR"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


class VolatilityLevel(Enum):
    """Volatility level"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskMode(Enum):
    """Risk management mode"""
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


class EventType(Enum):
    """System event type"""
    EMERGENCY_STOP = "EMERGENCY_STOP"
    EMERGENCY_STOP_CLEARED = "EMERGENCY_STOP_CLEARED"
    HEALTH_WARNING = "HEALTH_WARNING"
    HEALTH_RECOVERED = "HEALTH_RECOVERED"
    POLICY_CHANGE = "POLICY_CHANGE"
    ERROR = "ERROR"


@dataclass
class Trade:
    """
    Represents a completed trade with full lifecycle information.
    """
    id: str
    timestamp: datetime
    symbol: str
    strategy_id: str
    direction: TradeDirection
    
    # Entry
    entry_price: float
    entry_timestamp: datetime
    entry_size: float
    
    # Exit
    exit_price: float
    exit_timestamp: datetime
    exit_reason: TradeExitReason
    
    # Performance
    pnl: float
    pnl_pct: float
    r_multiple: float  # Risk-reward multiple
    
    # Context
    regime_at_entry: MarketRegime
    volatility_at_entry: VolatilityLevel
    risk_mode: RiskMode
    confidence: float
    
    # Costs
    commission: float = 0.0
    slippage: float = 0.0
    
    # Metadata
    model_version: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Trade duration in seconds"""
        return (self.exit_timestamp - self.entry_timestamp).total_seconds()
    
    @property
    def is_winner(self) -> bool:
        """Is this a winning trade"""
        return self.pnl > 0
    
    @property
    def gross_pnl(self) -> float:
        """PnL before costs"""
        return self.pnl + self.commission + self.slippage


@dataclass
class StrategyStats:
    """
    Performance statistics for a strategy over a time period.
    """
    strategy_id: str
    timestamp: datetime
    
    # Performance
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    
    # Risk
    max_drawdown: float
    avg_r_multiple: float
    profit_factor: float
    sharpe_ratio: float
    
    # Volume
    total_volume: float
    avg_trade_size: float
    
    # Activity
    active: bool
    signals_generated: int
    signals_accepted: int
    
    # Metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class SymbolStats:
    """
    Performance statistics for a symbol over a time period.
    """
    symbol: str
    timestamp: datetime
    
    # Performance
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    
    # Risk
    max_drawdown: float
    avg_r_multiple: float
    profit_factor: float
    
    # Volume
    total_volume: float
    
    # Market Context
    regime_distribution: dict[MarketRegime, int]
    volatility_distribution: dict[VolatilityLevel, int]


@dataclass
class EventLog:
    """
    System event log entry.
    """
    id: str
    timestamp: datetime
    event_type: EventType
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    description: str
    details: dict = field(default_factory=dict)
    
    # Context
    equity_at_event: Optional[float] = None
    drawdown_at_event: Optional[float] = None
    active_positions: int = 0


@dataclass
@dataclass
class EquityPoint:
    """
    Point on equity curve.
    """
    timestamp: datetime
    equity: float
    balance: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class DrawdownPeriod:
    """
    Drawdown period with start, bottom, and recovery.
    """
    start_timestamp: datetime
    bottom_timestamp: datetime
    recovery_timestamp: Optional[datetime]
    
    peak_equity: float
    bottom_equity: float
    recovery_equity: Optional[float]
    
    max_drawdown: float
    max_drawdown_pct: float
    duration_days: float
    recovery_days: Optional[float]
    
    @property
    def recovered(self) -> bool:
        """Has drawdown recovered"""
        return self.recovery_timestamp is not None


@dataclass
class PerformanceSummary:
    """
    Summary statistics for a performance analysis.
    """
    # Time period
    start_date: datetime
    end_date: datetime
    days: int
    
    # Performance
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_pnl_pct: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_r_multiple: float
    
    # Best/Worst
    best_trade_pnl: float
    worst_trade_pnl: float
    best_day_pnl: float
    worst_day_pnl: float
    
    # Streaks
    longest_win_streak: int
    longest_loss_streak: int
    current_streak: int
    current_streak_type: str  # "WIN" or "LOSS"

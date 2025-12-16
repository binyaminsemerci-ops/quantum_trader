"""
Data models for scenario and stress testing
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from enum import Enum


class ScenarioType(str, Enum):
    """Types of stress test scenarios"""
    HISTORIC_REPLAY = "historic_replay"
    FLASH_CRASH = "flash_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    TREND_SHIFT = "trend_shift"
    LIQUIDITY_DROP = "liquidity_drop"
    SPREAD_EXPLOSION = "spread_explosion"
    DATA_CORRUPTION = "data_corruption"
    MODEL_FAILURE = "model_failure"
    EXECUTION_FAILURE = "execution_failure"
    LATENCY_SPIKE = "latency_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    PUMP_DUMP = "pump_dump"
    MIXED_CUSTOM = "mixed_custom"


@dataclass
class Scenario:
    """
    Defines a stress test scenario.
    
    Attributes:
        name: Human-readable scenario name
        type: Scenario type (historic replay, synthetic stress, etc.)
        parameters: Type-specific parameters (drop_pct, multiplier, etc.)
        start: Start datetime for historical replays
        end: End datetime for historical replays
        seed: Random seed for reproducibility
        symbols: Symbols to test (defaults to BTC/ETH if None)
        description: Optional detailed description
    """
    name: str
    type: ScenarioType
    parameters: dict[str, Any] = field(default_factory=dict)
    start: datetime | None = None
    end: datetime | None = None
    seed: int | None = None
    symbols: list[str] | None = None
    description: str = ""
    
    def __post_init__(self):
        """Set default symbols if not provided"""
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT"]


@dataclass
class ExecutionResult:
    """
    Result of a single simulated order execution.
    
    Attributes:
        success: Whether execution succeeded
        filled_price: Actual fill price (with slippage)
        filled_qty: Actual quantity filled
        slippage_pct: Slippage percentage
        latency_ms: Simulated execution latency
        error_reason: Failure reason if not successful
        timestamp: Execution timestamp
    """
    success: bool
    filled_price: float
    filled_qty: float
    slippage_pct: float
    latency_ms: float
    error_reason: str | None = None
    timestamp: datetime | None = None


@dataclass
class TradeRecord:
    """
    Record of a simulated trade.
    
    Attributes:
        symbol: Trading pair
        side: BUY or SELL
        entry_price: Entry price
        exit_price: Exit price (if closed)
        qty: Quantity
        pnl: Realized PnL
        strategy: Strategy that generated signal
        entry_time: Entry timestamp
        exit_time: Exit timestamp (if closed)
        closed: Whether trade is closed
        exit_reason: Reason for exit (TP/SL/timeout/emergency)
    """
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None
    qty: float
    pnl: float
    strategy: str
    entry_time: datetime
    exit_time: datetime | None = None
    closed: bool = False
    exit_reason: str | None = None


@dataclass
class ScenarioResult:
    """
    Complete results of a scenario stress test.
    
    Attributes:
        scenario_name: Name of tested scenario
        pnl_curve: Time series of cumulative PnL
        equity_curve: Time series of equity values
        max_drawdown: Maximum drawdown percentage
        max_drawdown_duration: Longest drawdown duration (bars)
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        winrate: Win rate percentage
        profit_factor: Profit factor (gross profit / gross loss)
        sharpe_ratio: Sharpe ratio
        trades: List of all trade records
        emergency_stops: Number of ESS activations
        failed_models: List of models that failed
        failed_strategies: List of strategies that failed
        policy_transitions: List of MSC policy changes
        opportunity_scores: Time series of opportunity scores
        regime_distribution: Count of bars in each regime
        execution_failures: Count of failed executions
        data_quality_issues: Count of data corruption events
        latency_spikes: Count of latency spike events
        notes: Additional observations and warnings
        start_time: Simulation start time
        end_time: Simulation end time
        duration_seconds: Total simulation duration
        success: Whether test completed successfully
    """
    scenario_name: str
    pnl_curve: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    winrate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    emergency_stops: int = 0
    failed_models: list[str] = field(default_factory=list)
    failed_strategies: list[str] = field(default_factory=list)
    policy_transitions: list[dict[str, Any]] = field(default_factory=list)
    opportunity_scores: list[float] = field(default_factory=list)
    regime_distribution: dict[str, int] = field(default_factory=dict)
    execution_failures: int = 0
    data_quality_issues: int = 0
    latency_spikes: int = 0
    notes: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    success: bool = True
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_trades > 0:
            self.winrate = (self.winning_trades / self.total_trades) * 100
        
        if self.losing_trades > 0:
            gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
            if gross_loss > 0:
                self.profit_factor = gross_profit / gross_loss
    
    def summary(self) -> dict[str, Any]:
        """Return summary statistics"""
        return {
            "scenario": self.scenario_name,
            "success": self.success,
            "total_trades": self.total_trades,
            "winrate": f"{self.winrate:.1f}%",
            "max_drawdown": f"{self.max_drawdown:.2f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "emergency_stops": self.emergency_stops,
            "failed_models": len(self.failed_models),
            "failed_strategies": len(self.failed_strategies),
            "execution_failures": self.execution_failures,
            "duration": f"{self.duration_seconds:.1f}s"
        }

"""
Replay Result - Contains output of a replay session
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TradeRecord:
    """Record of a single trade executed during replay"""
    timestamp: datetime
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    strategy_id: Optional[str] = None
    confidence: float = 0.0
    regime: Optional[str] = None
    
    # Execution details
    slippage: float = 0.0
    commission: float = 0.0
    execution_time_ms: float = 0.0
    
    # Exit details
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "TP", "SL", "MANUAL", "ESS"
    
    # Context
    equity_before: float = 0.0
    equity_after: float = 0.0
    position_size_pct: float = 0.0
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    """Record of a system event during replay"""
    timestamp: datetime
    event_type: str  # "MSC_UPDATE", "ESS_TRIGGER", "RISK_BREACH", etc.
    description: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolStats:
    """Performance statistics for a single symbol"""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_bars_held: float = 0.0


@dataclass
class StrategyStats:
    """Performance statistics for a single strategy"""
    strategy_id: str
    total_signals: int = 0
    accepted_signals: int = 0
    rejected_signals: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class ReplayResult:
    """
    Complete output of a replay session.
    
    Contains all trades, events, performance metrics, and analysis results.
    """
    # Configuration
    config: "ReplayConfig"
    
    # Execution info
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_candles_processed: int
    
    # Performance metrics
    initial_balance: float
    final_balance: float
    pnl_total: float
    pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_pnl: float
    avg_trade_duration_bars: float
    
    # Equity curve
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    
    # Trade records
    trades: list[TradeRecord] = field(default_factory=list)
    
    # Event log
    events: list[EventRecord] = field(default_factory=list)
    
    # Per-symbol breakdown
    per_symbol_stats: dict[str, SymbolStats] = field(default_factory=dict)
    
    # Per-strategy breakdown
    per_strategy_stats: dict[str, StrategyStats] = field(default_factory=dict)
    
    # System events
    triggered_emergency_stops: int = 0
    policy_changes: int = 0
    risk_breaches: int = 0
    
    # Analysis notes
    notes: list[str] = field(default_factory=list)
    
    # Raw data (optional, for detailed analysis)
    raw_equity_curve: Optional[list[float]] = None
    raw_timestamps: Optional[list[datetime]] = None
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"=== Replay Summary ===",
            f"Period: {self.config.start} to {self.config.end}",
            f"Duration: {self.duration_seconds:.1f}s ({self.total_candles_processed} candles)",
            f"Symbols: {', '.join(self.config.symbols)}",
            f"Mode: {self.config.mode.value}",
            f"",
            f"Performance:",
            f"  Initial Balance: ${self.initial_balance:,.2f}",
            f"  Final Balance: ${self.final_balance:,.2f}",
            f"  Total PnL: ${self.pnl_total:,.2f} ({self.pnl_pct:.2f}%)",
            f"  Max Drawdown: {self.max_drawdown_pct:.2f}%",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Profit Factor: {self.profit_factor:.2f}",
            f"",
            f"Trading:",
            f"  Total Trades: {self.total_trades}",
            f"  Win Rate: {self.win_rate:.1f}%",
            f"  Winning: {self.winning_trades}",
            f"  Losing: {self.losing_trades}",
            f"  Avg Trade PnL: ${self.avg_trade_pnl:.2f}",
            f"",
            f"System Events:",
            f"  Emergency Stops: {self.triggered_emergency_stops}",
            f"  Policy Changes: {self.policy_changes}",
            f"  Risk Breaches: {self.risk_breaches}",
        ]
        
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        
        return "\n".join(lines)
    
    def get_best_symbol(self) -> Optional[str]:
        """Get symbol with highest PnL"""
        if not self.per_symbol_stats:
            return None
        return max(self.per_symbol_stats.items(), key=lambda x: x[1].total_pnl)[0]
    
    def get_best_strategy(self) -> Optional[str]:
        """Get strategy with highest PnL"""
        if not self.per_strategy_stats:
            return None
        return max(self.per_strategy_stats.items(), key=lambda x: x[1].total_pnl)[0]
    
    def get_worst_drawdown_period(self) -> Optional[tuple[datetime, datetime]]:
        """Find period with worst drawdown"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return None
        
        peak_equity = self.equity_curve[0][1]
        peak_time = self.equity_curve[0][0]
        worst_dd = 0.0
        worst_period = None
        
        for timestamp, equity in self.equity_curve:
            if equity > peak_equity:
                peak_equity = equity
                peak_time = timestamp
            else:
                dd = (peak_equity - equity) / peak_equity * 100
                if dd > worst_dd:
                    worst_dd = dd
                    worst_period = (peak_time, timestamp)
        
        return worst_period

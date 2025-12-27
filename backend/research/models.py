"""
Data models for Strategy Generator AI.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum


class StrategyStatus(str, Enum):
    """Lifecycle status of a trading strategy"""
    CANDIDATE = "CANDIDATE"  # Just created, backtest only
    SHADOW = "SHADOW"        # Forward testing with paper trades
    LIVE = "LIVE"            # Actively trading real money
    DISABLED = "DISABLED"    # Underperforming, not used


class RegimeFilter(str, Enum):
    """Market regime classification for strategy filtering"""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    ANY = "ANY"


@dataclass
class StrategyConfig:
    """
    Complete specification of a trading strategy.
    
    This defines all parameters needed to execute a strategy:
    filters, entry/exit logic, risk parameters, and metadata.
    """
    strategy_id: str
    name: str
    
    # === Filters ===
    regime_filter: RegimeFilter = RegimeFilter.ANY
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["15m"])
    min_confidence: float = 0.60
    
    # === Entry Rules ===
    entry_type: str = "ENSEMBLE_CONSENSUS"  # MOMENTUM | MEAN_REVERSION | BREAKOUT
    entry_params: dict[str, Any] = field(default_factory=dict)
    
    # === Exit Rules ===
    tp_percent: float = 0.016  # 1.6%
    sl_percent: float = 0.008  # 0.8%
    use_trailing: bool = True
    trailing_callback: float = 0.015  # 1.5%
    
    # === Risk Parameters ===
    max_risk_per_trade: float = 0.02  # 2% of equity
    max_leverage: float = 10.0
    max_concurrent_positions: int = 5
    
    # === Metadata ===
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: StrategyStatus = StrategyStatus.CANDIDATE
    generation: int = 0  # Evolution generation number
    parent_ids: list[str] = field(default_factory=list)  # For tracking lineage
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "regime_filter": self.regime_filter.value,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "min_confidence": self.min_confidence,
            "entry_type": self.entry_type,
            "entry_params": self.entry_params,
            "tp_percent": self.tp_percent,
            "sl_percent": self.sl_percent,
            "use_trailing": self.use_trailing,
            "trailing_callback": self.trailing_callback,
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_leverage": self.max_leverage,
            "max_concurrent_positions": self.max_concurrent_positions,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


@dataclass
class StrategyStats:
    """
    Performance metrics for a strategy over a specific period.
    
    Can represent backtest results, shadow testing performance,
    or live trading results.
    """
    strategy_id: str
    source: str  # "BACKTEST" | "SHADOW" | "LIVE"
    
    # === Period ===
    start_date: datetime
    end_date: datetime
    timestamp: datetime = field(default_factory=datetime.utcnow)  # When stats were recorded
    
    # === Trade Statistics ===
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # === Performance Metrics ===
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    
    # === Risk Metrics ===
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    # === Average Metrics ===
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr_ratio: float = 0.0
    
    # === Execution ===
    avg_bars_in_trade: float = 0.0
    
    # === Fitness ===
    fitness_score: float = 0.0
    
    def calculate_derived_metrics(self) -> None:
        """Calculate metrics derived from raw trade data"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            
        if self.winning_trades > 0:
            self.avg_win = self.gross_profit / self.winning_trades
            
        if self.losing_trades > 0:
            self.avg_loss = abs(self.gross_loss) / self.losing_trades
            
        if self.gross_loss != 0:
            self.profit_factor = self.gross_profit / abs(self.gross_loss)
        else:
            self.profit_factor = float('inf') if self.gross_profit > 0 else 0.0
            
        if self.avg_loss != 0:
            self.avg_rr_ratio = self.avg_win / self.avg_loss
        
        # Composite fitness score
        # Balance: profit factor, win rate, drawdown, sample size
        self.fitness_score = self._calculate_fitness()
    
    def _calculate_fitness(self) -> float:
        """
        Calculate composite fitness score for strategy ranking.
        
        Higher is better. Balances multiple metrics:
        - Profit factor (primary)
        - Win rate (secondary)
        - Max drawdown (penalty)
        - Sample size (confidence)
        """
        if self.total_trades < 30:
            # Insufficient sample size penalty
            sample_penalty = self.total_trades / 30.0
        else:
            sample_penalty = 1.0
        
        # Base score from profit factor
        pf_score = min(self.profit_factor, 5.0) / 5.0  # Normalize to [0, 1]
        
        # Win rate contribution
        wr_score = self.win_rate
        
        # Drawdown penalty (lower is better)
        dd_penalty = max(0, 1.0 - (self.max_drawdown_pct / 0.30))  # 30% DD = 0 score
        
        # Weighted composite
        fitness = (
            pf_score * 0.40 +
            wr_score * 0.20 +
            dd_penalty * 0.20 +
            sample_penalty * 0.20
        )
        
        return fitness * 100.0  # Scale to 0-100
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage"""
        return {
            "strategy_id": self.strategy_id,
            "source": self.source,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_rr_ratio": self.avg_rr_ratio,
            "avg_bars_in_trade": self.avg_bars_in_trade,
            "fitness_score": self.fitness_score,
        }

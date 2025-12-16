"""
Replay Configuration - Defines parameters for replay sessions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ReplayMode(str, Enum):
    """Replay execution mode"""
    FULL = "full"                      # Full system replay
    STRATEGY_ONLY = "strategy_only"    # Only strategy runtime + decisions
    MODEL_ONLY = "model_only"          # Only model predictions vs reality
    EXECUTION_ONLY = "execution_only"  # Only trade log validation


@dataclass
class ReplayConfig:
    """
    Configuration for a replay session.
    
    Attributes:
        start: Start timestamp for replay
        end: End timestamp for replay
        symbols: List of symbols to replay
        timeframe: Candle timeframe (e.g., "1m", "5m", "1h")
        mode: Replay mode (full/strategy_only/model_only/execution_only)
        initial_balance: Starting balance for replay
        speed: Replay speed (0.0 = as fast as possible, >0 = delay in seconds)
        include_msc: Whether to include Meta Strategy Controller updates
        include_ess: Whether to simulate Emergency Stop System
        strategy_ids: Optional list of specific strategies to test
        enable_logging: Whether to log detailed events
        save_equity_curve: Whether to save full equity curve
        save_trades: Whether to save all trade records
    """
    start: datetime
    end: datetime
    symbols: list[str]
    timeframe: str
    
    # Execution mode
    mode: ReplayMode = ReplayMode.FULL
    
    # Initial conditions
    initial_balance: float = 10_000.0
    
    # Replay speed control
    speed: float = 0.0  # 0.0 = as fast as possible
    
    # System components
    include_msc: bool = True
    include_ess: bool = True
    
    # Filtering
    strategy_ids: Optional[list[str]] = None
    
    # Output options
    enable_logging: bool = True
    save_equity_curve: bool = True
    save_trades: bool = True
    
    # Advanced options
    slippage_model: str = "realistic"  # "none", "realistic", "pessimistic"
    commission_rate: float = 0.001     # 0.1% default
    max_trades_per_bar: int = 10       # Limit trades per candle
    
    def __post_init__(self):
        """Validate configuration"""
        if self.start >= self.end:
            raise ValueError("start must be before end")
        
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
        
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        
        if self.speed < 0:
            raise ValueError("speed must be non-negative")
        
        if self.timeframe not in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]:
            raise ValueError(f"Invalid timeframe: {self.timeframe}")
    
    def duration_days(self) -> float:
        """Calculate replay duration in days"""
        delta = self.end - self.start
        return delta.total_seconds() / 86400
    
    def estimated_candles(self) -> int:
        """Estimate total number of candles to process"""
        duration_minutes = (self.end - self.start).total_seconds() / 60
        
        # Map timeframe to minutes
        timeframe_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "1d": 1440
        }
        
        tf_mins = timeframe_minutes.get(self.timeframe, 60)
        return int(duration_minutes / tf_mins) * len(self.symbols)

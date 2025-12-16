"""Symbol Performance Manager - Per-symbol stats tracking and risk adjustment."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _parse_float(value: str | None, *, default: float) -> float:
    """Parse float from environment variable."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, *, default: int) -> int:
    """Parse int from environment variable."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class SymbolPerformanceConfig:
    """Configuration for symbol performance tracking."""
    
    # Minimum trades before adjusting risk
    min_trades_for_adjustment: int = 5
    
    # Performance thresholds
    poor_winrate_threshold: float = 0.30      # <30% winrate = poor
    good_winrate_threshold: float = 0.55      # >55% winrate = good
    poor_avg_R_threshold: float = 0.0         # <0R average = poor
    good_avg_R_threshold: float = 1.5         # >1.5R average = good
    
    # Risk adjustment multipliers
    poor_risk_multiplier: float = 0.5         # Reduce risk to 50%
    good_risk_multiplier: float = 1.0         # Keep at 100% (never increase)
    
    # Disable threshold
    disable_after_losses: int = 10            # Disable after 10 consecutive losses
    reenable_after_wins: int = 3              # Re-enable after 3 wins
    
    # Persistence
    persistence_file: Optional[str] = "data/symbol_performance.json"
    
    @classmethod
    def from_env(cls) -> SymbolPerformanceConfig:
        """Load configuration from environment variables."""
        return cls(
            min_trades_for_adjustment=_parse_int(
                os.getenv("PERF_MIN_TRADES"),
                default=5
            ),
            poor_winrate_threshold=_parse_float(
                os.getenv("PERF_POOR_WINRATE"),
                default=0.30
            ),
            good_winrate_threshold=_parse_float(
                os.getenv("PERF_GOOD_WINRATE"),
                default=0.55
            ),
            poor_avg_R_threshold=_parse_float(
                os.getenv("PERF_POOR_AVG_R"),
                default=0.0
            ),
            good_avg_R_threshold=_parse_float(
                os.getenv("PERF_GOOD_AVG_R"),
                default=1.5
            ),
            poor_risk_multiplier=_parse_float(
                os.getenv("PERF_POOR_RISK_MULT"),
                default=0.5
            ),
            good_risk_multiplier=_parse_float(
                os.getenv("PERF_GOOD_RISK_MULT"),
                default=1.0
            ),
            disable_after_losses=_parse_int(
                os.getenv("PERF_DISABLE_AFTER_LOSSES"),
                default=10
            ),
            reenable_after_wins=_parse_int(
                os.getenv("PERF_REENABLE_AFTER_WINS"),
                default=3
            ),
            persistence_file=os.getenv("PERF_PERSISTENCE_FILE", "data/symbol_performance.json")
        )


@dataclass
class SymbolStats:
    """Performance statistics for a single symbol."""
    symbol: str
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    total_R: float = 0.0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    is_enabled: bool = True
    last_updated: Optional[str] = None
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.trade_count == 0:
            return 0.0
        return self.wins / self.trade_count
    
    @property
    def avg_R(self) -> float:
        """Calculate average R-multiple."""
        if self.trade_count == 0:
            return 0.0
        return self.total_R / self.trade_count
    
    @property
    def avg_pnl(self) -> float:
        """Calculate average PnL per trade."""
        if self.trade_count == 0:
            return 0.0
        return self.total_pnl / self.trade_count
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> SymbolStats:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TradeResult:
    """Result of a completed trade."""
    symbol: str
    pnl: float              # Dollar PnL
    R_multiple: float       # R-multiple (e.g., 2.5 for 2.5R win, -1.0 for 1R loss)
    was_winner: bool
    timestamp: Optional[datetime] = None


class SymbolPerformanceManager:
    """
    Track per-symbol performance and adjust risk based on historical results.
    
    Maintains statistics for each symbol including:
    - Win rate
    - Average R-multiple
    - Cumulative PnL
    - Consecutive wins/losses
    
    Uses performance data to:
    - Reduce risk on poorly performing symbols
    - Disable symbols after extended losing streaks
    - Re-enable symbols after recovery
    """
    
    def __init__(self, config: Optional[SymbolPerformanceConfig] = None):
        """
        Initialize symbol performance manager.
        
        Args:
            config: Performance configuration. If None, loads from environment.
        """
        self.config = config or SymbolPerformanceConfig.from_env()
        self.stats: Dict[str, SymbolStats] = {}
        
        # Load persisted stats if available
        self._load_stats()
        
        logger.info(
            f"[OK] SymbolPerformanceManager initialized: "
            f"Min trades={self.config.min_trades_for_adjustment}, "
            f"Poor WR<{self.config.poor_winrate_threshold:.0%}, "
            f"Disable after {self.config.disable_after_losses} losses"
        )
    
    def update_stats(self, trade_result: TradeResult) -> None:
        """
        Update statistics with a completed trade.
        
        Args:
            trade_result: Result of completed trade
        """
        symbol = trade_result.symbol
        
        # Initialize stats if not exists
        if symbol not in self.stats:
            self.stats[symbol] = SymbolStats(symbol=symbol)
        
        stats = self.stats[symbol]
        
        # Update counts
        stats.trade_count += 1
        if trade_result.was_winner:
            stats.wins += 1
            stats.consecutive_wins += 1
            stats.consecutive_losses = 0
        else:
            stats.losses += 1
            stats.consecutive_losses += 1
            stats.consecutive_wins = 0
        
        # Update totals
        stats.total_R += trade_result.R_multiple
        stats.total_pnl += trade_result.pnl
        stats.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Check if symbol should be disabled
        if stats.consecutive_losses >= self.config.disable_after_losses:
            stats.is_enabled = False
            logger.warning(
                f"[WARNING] {symbol} DISABLED after {stats.consecutive_losses} consecutive losses"
            )
        
        # Check if symbol should be re-enabled
        if not stats.is_enabled and stats.consecutive_wins >= self.config.reenable_after_wins:
            stats.is_enabled = True
            logger.info(
                f"[OK] {symbol} RE-ENABLED after {stats.consecutive_wins} consecutive wins"
            )
        
        logger.debug(
            f"[CHART] {symbol} Stats Updated: "
            f"{stats.trade_count} trades, "
            f"WR={stats.win_rate:.1%}, "
            f"Avg R={stats.avg_R:.2f}, "
            f"{'ENABLED' if stats.is_enabled else 'DISABLED'}"
        )
        
        # Persist stats
        self._save_stats()
    
    def get_risk_modifier(self, symbol: str) -> float:
        """
        Get risk multiplier for a symbol based on performance.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Risk multiplier (0.5 to 1.0). Multiply base risk by this value.
        """
        if symbol not in self.stats:
            # No history - use default (1.0)
            return 1.0
        
        stats = self.stats[symbol]
        
        # Need minimum trades before adjusting
        if stats.trade_count < self.config.min_trades_for_adjustment:
            return 1.0
        
        # Symbol disabled - return 0 (no trading)
        if not stats.is_enabled:
            return 0.0
        
        # Check for poor performance
        if (stats.win_rate < self.config.poor_winrate_threshold or 
            stats.avg_R < self.config.poor_avg_R_threshold):
            logger.debug(
                f"🔽 {symbol} Poor performance: "
                f"WR={stats.win_rate:.1%}, Avg R={stats.avg_R:.2f} "
                f"→ Risk multiplier={self.config.poor_risk_multiplier:.1f}x"
            )
            return self.config.poor_risk_multiplier
        
        # Check for good performance (but never increase risk)
        if (stats.win_rate >= self.config.good_winrate_threshold and 
            stats.avg_R >= self.config.good_avg_R_threshold):
            logger.debug(
                f"[CHART_UP] {symbol} Good performance: "
                f"WR={stats.win_rate:.1%}, Avg R={stats.avg_R:.2f} "
                f"→ Risk multiplier={self.config.good_risk_multiplier:.1f}x"
            )
            return self.config.good_risk_multiplier
        
        # Average performance - standard risk
        return 1.0
    
    def should_trade_symbol(self, symbol: str) -> bool:
        """
        Check if symbol should be traded based on performance.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            True if symbol can be traded, False if disabled
        """
        if symbol not in self.stats:
            return True  # No history - allow trading
        
        return self.stats[symbol].is_enabled
    
    def get_stats(self, symbol: str) -> Optional[SymbolStats]:
        """
        Get statistics for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            SymbolStats if available, None otherwise
        """
        return self.stats.get(symbol)
    
    def get_all_stats(self) -> Dict[str, SymbolStats]:
        """Get statistics for all symbols."""
        return self.stats.copy()
    
    def reset_symbol(self, symbol: str) -> None:
        """
        Reset statistics for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.stats:
            self.stats[symbol] = SymbolStats(symbol=symbol)
            logger.info(f"🔄 {symbol} stats reset")
            self._save_stats()
    
    def _load_stats(self) -> None:
        """Load persisted statistics from file."""
        if not self.config.persistence_file:
            return
        
        try:
            path = Path(self.config.persistence_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    for symbol, stats_dict in data.items():
                        self.stats[symbol] = SymbolStats.from_dict(stats_dict)
                logger.info(f"[OK] Loaded stats for {len(self.stats)} symbols from {path}")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to load stats: {e}")
    
    def _save_stats(self) -> None:
        """Persist statistics to file."""
        if not self.config.persistence_file:
            return
        
        try:
            path = Path(self.config.persistence_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {symbol: stats.to_dict() for symbol, stats in self.stats.items()}
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"💾 Saved stats for {len(self.stats)} symbols to {path}")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to save stats: {e}")

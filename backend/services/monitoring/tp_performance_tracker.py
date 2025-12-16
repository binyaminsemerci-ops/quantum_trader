"""
TP Performance Tracking Module
===============================

Tracks take-profit performance metrics for continuous improvement.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TPMetrics:
    """TP performance metrics for a strategy/symbol."""
    strategy_id: str
    symbol: str
    
    # Hit rate metrics
    tp_attempts: int = 0
    tp_hits: int = 0
    tp_misses: int = 0
    tp_hit_rate: float = 0.0
    
    # Slippage metrics
    total_slippage_pct: float = 0.0
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    
    # Timing metrics
    avg_time_to_tp_minutes: float = 0.0
    fastest_tp_minutes: float = float('inf')
    slowest_tp_minutes: float = 0.0
    
    # Profit metrics
    total_tp_profit_usd: float = 0.0
    avg_tp_profit_usd: float = 0.0
    
    # Premature exit tracking
    premature_exits: int = 0  # Exited before TP but could have hit
    missed_opportunities_usd: float = 0.0
    
    # Last updated
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TPPerformanceTracker:
    """
    Tracks TP performance across all trades for continuous learning.
    
    Features:
    - Hit rate tracking per strategy/symbol
    - Slippage measurement
    - Time-to-TP statistics
    - Premature exit detection
    - Performance feedback to RL training
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize TP Performance Tracker.
        
        Args:
            storage_path: Path to persist metrics (default: /app/tmp/tp_metrics.json)
        """
        self.logger = logging.getLogger(__name__)
        
        self.storage_path = storage_path or Path("/app/tmp/tp_metrics.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage: {(strategy_id, symbol): TPMetrics}
        self.metrics: Dict[tuple, TPMetrics] = {}
        
        # Load existing metrics
        self._load_metrics()
        
        self.logger.info(f"[TP Tracker] Initialized with {len(self.metrics)} tracked pairs")
    
    def record_tp_attempt(
        self,
        strategy_id: str,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        tp_target_price: float,
        side: str
    ) -> str:
        """
        Record a new TP attempt.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_time: Position entry time
            entry_price: Entry price
            tp_target_price: TP target price
            side: Position side (LONG/SHORT)
            
        Returns:
            Tracking ID for this TP attempt
        """
        key = (strategy_id, symbol)
        if key not in self.metrics:
            self.metrics[key] = TPMetrics(strategy_id=strategy_id, symbol=symbol)
        
        self.metrics[key].tp_attempts += 1
        self._save_metrics()
        
        # Generate tracking ID
        tracking_id = f"{strategy_id}_{symbol}_{entry_time.timestamp()}"
        return tracking_id
    
    def record_tp_hit(
        self,
        strategy_id: str,
        symbol: str,
        exit_time: datetime,
        exit_price: float,
        tp_target_price: float,
        entry_time: datetime,
        entry_price: float,
        profit_usd: float
    ):
        """
        Record a successful TP hit.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            exit_time: Exit time
            exit_price: Actual exit price
            tp_target_price: Target TP price
            entry_time: Entry time
            entry_price: Entry price
            profit_usd: Profit in USD
        """
        key = (strategy_id, symbol)
        if key not in self.metrics:
            self.metrics[key] = TPMetrics(strategy_id=strategy_id, symbol=symbol)
        
        metrics = self.metrics[key]
        metrics.tp_hits += 1
        
        # Calculate slippage
        slippage_pct = abs(exit_price - tp_target_price) / tp_target_price
        metrics.total_slippage_pct += slippage_pct
        metrics.avg_slippage_pct = metrics.total_slippage_pct / metrics.tp_hits
        metrics.max_slippage_pct = max(metrics.max_slippage_pct, slippage_pct)
        
        # Calculate time to TP
        time_to_tp = (exit_time - entry_time).total_seconds() / 60.0  # minutes
        if metrics.tp_hits == 1:
            metrics.avg_time_to_tp_minutes = time_to_tp
        else:
            # Running average
            metrics.avg_time_to_tp_minutes = (
                (metrics.avg_time_to_tp_minutes * (metrics.tp_hits - 1) + time_to_tp)
                / metrics.tp_hits
            )
        metrics.fastest_tp_minutes = min(metrics.fastest_tp_minutes, time_to_tp)
        metrics.slowest_tp_minutes = max(metrics.slowest_tp_minutes, time_to_tp)
        
        # Track profit
        metrics.total_tp_profit_usd += profit_usd
        metrics.avg_tp_profit_usd = metrics.total_tp_profit_usd / metrics.tp_hits
        
        # Update hit rate
        total = metrics.tp_hits + metrics.tp_misses
        metrics.tp_hit_rate = metrics.tp_hits / total if total > 0 else 0.0
        
        metrics.last_updated = datetime.now(timezone.utc)
        
        self._save_metrics()
        
        self.logger.info(
            f"[TP Tracker] Hit recorded: {strategy_id}/{symbol} "
            f"(rate={metrics.tp_hit_rate:.1%}, slippage={slippage_pct:.3%}, "
            f"time={time_to_tp:.1f}min)"
        )
    
    def record_tp_miss(
        self,
        strategy_id: str,
        symbol: str,
        exit_reason: str,
        exit_price: float,
        tp_target_price: float,
        premature: bool = False,
        missed_profit_usd: float = 0.0
    ):
        """
        Record a TP miss (position closed without hitting TP).
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            exit_reason: Why position was closed
            exit_price: Exit price
            tp_target_price: Target TP price
            premature: True if TP would have hit if held longer
            missed_profit_usd: Potential profit if had waited for TP
        """
        key = (strategy_id, symbol)
        if key not in self.metrics:
            self.metrics[key] = TPMetrics(strategy_id=strategy_id, symbol=symbol)
        
        metrics = self.metrics[key]
        metrics.tp_misses += 1
        
        if premature:
            metrics.premature_exits += 1
            metrics.missed_opportunities_usd += missed_profit_usd
        
        # Update hit rate
        total = metrics.tp_hits + metrics.tp_misses
        metrics.tp_hit_rate = metrics.tp_hits / total if total > 0 else 0.0
        
        metrics.last_updated = datetime.now(timezone.utc)
        
        self._save_metrics()
        
        self.logger.info(
            f"[TP Tracker] Miss recorded: {strategy_id}/{symbol} "
            f"(reason={exit_reason}, premature={premature}, rate={metrics.tp_hit_rate:.1%})"
        )
    
    def get_metrics(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[TPMetrics]:
        """
        Get TP metrics filtered by strategy and/or symbol.
        
        Args:
            strategy_id: Filter by strategy (None = all)
            symbol: Filter by symbol (None = all)
            
        Returns:
            List of TPMetrics matching filters
        """
        results = []
        for (strat, sym), metrics in self.metrics.items():
            if strategy_id and strat != strategy_id:
                continue
            if symbol and sym != symbol:
                continue
            results.append(metrics)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall TP performance summary.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.metrics:
            return {
                'total_attempts': 0,
                'total_hits': 0,
                'total_misses': 0,
                'overall_hit_rate': 0.0,
                'avg_slippage': 0.0,
                'total_profit': 0.0,
                'tracked_pairs': 0,
                'premature_exits': 0,
                'missed_opportunities_usd': 0.0
            }
        
        total_attempts = sum(m.tp_attempts for m in self.metrics.values())
        total_hits = sum(m.tp_hits for m in self.metrics.values())
        total_misses = sum(m.tp_misses for m in self.metrics.values())
        
        # Weighted average slippage
        total_slippage = sum(m.total_slippage_pct for m in self.metrics.values())
        avg_slippage = total_slippage / total_hits if total_hits > 0 else 0.0
        
        total_profit = sum(m.total_tp_profit_usd for m in self.metrics.values())
        
        return {
            'total_attempts': total_attempts,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            'avg_slippage': avg_slippage,
            'total_profit': total_profit,
            'tracked_pairs': len(self.metrics),
            'premature_exits': sum(m.premature_exits for m in self.metrics.values()),
            'missed_opportunities_usd': sum(m.missed_opportunities_usd for m in self.metrics.values())
        }
    
    def get_feedback_for_rl_training(self) -> Dict[str, float]:
        """
        Generate feedback metrics for RL training.
        
        Returns:
            Dictionary of metrics to include in RL reward function
        """
        summary = self.get_summary()
        
        return {
            'tp_hit_rate': summary['overall_hit_rate'],
            'avg_slippage': summary['avg_slippage'],
            'premature_exit_rate': (
                summary['premature_exits'] / summary['total_attempts']
                if summary['total_attempts'] > 0 else 0.0
            )
        }
    
    def get_strategy_tp_feedback(
        self,
        strategy_id: str,
        symbol: Optional[str] = None,
        min_attempts: int = 10
    ) -> Optional[Dict[str, float]]:
        """
        Get TP performance feedback for a specific strategy/symbol.
        
        Used by CLM v3 to compute TP reward weight for RL training.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Optional symbol (if None, aggregates all symbols for strategy)
            min_attempts: Minimum attempts required for valid feedback
        
        Returns:
            Dictionary with tp_hit_rate, avg_r_multiple, total_attempts
            or None if insufficient data
        """
        # Filter metrics for this strategy
        if symbol:
            # Specific pair
            metrics_list = [m for (s, sym), m in self.metrics.items() 
                          if s == strategy_id and sym == symbol]
        else:
            # All symbols for strategy
            metrics_list = [m for (s, sym), m in self.metrics.items() 
                          if s == strategy_id]
        
        if not metrics_list:
            return None
        
        # Aggregate metrics
        total_attempts = sum(m.tp_attempts for m in metrics_list)
        total_hits = sum(m.tp_hits for m in metrics_list)
        total_misses = sum(m.tp_misses for m in metrics_list)
        
        if total_attempts < min_attempts:
            return None
        
        # Calculate hit rate
        tp_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        
        # Estimate avg R multiple using inverse relationship: R ≈ 1 / hit_rate
        # This matches the TPOptimizerV3 calculation
        avg_r_multiple = 1.0 / max(tp_hit_rate, 0.05) if tp_hit_rate > 0 else 1.5
        avg_r_multiple = min(avg_r_multiple, 5.0)  # Cap at 5R
        
        return {
            'tp_hit_rate': tp_hit_rate,
            'avg_r_multiple': avg_r_multiple,
            'total_attempts': total_attempts,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'premature_exit_rate': (
                sum(m.premature_exits for m in metrics_list) / total_attempts
                if total_attempts > 0 else 0.0
            )
        }
    
    def _save_metrics(self):
        """Persist metrics to disk."""
        try:
            data = {}
            for (strat, sym), metrics in self.metrics.items():
                key = f"{strat}_{sym}"
                data[key] = {
                    'strategy_id': metrics.strategy_id,
                    'symbol': metrics.symbol,
                    'tp_attempts': metrics.tp_attempts,
                    'tp_hits': metrics.tp_hits,
                    'tp_misses': metrics.tp_misses,
                    'tp_hit_rate': metrics.tp_hit_rate,
                    'total_slippage_pct': metrics.total_slippage_pct,
                    'avg_slippage_pct': metrics.avg_slippage_pct,
                    'max_slippage_pct': metrics.max_slippage_pct,
                    'avg_time_to_tp_minutes': metrics.avg_time_to_tp_minutes,
                    'fastest_tp_minutes': metrics.fastest_tp_minutes if metrics.fastest_tp_minutes != float('inf') else 0,
                    'slowest_tp_minutes': metrics.slowest_tp_minutes,
                    'total_tp_profit_usd': metrics.total_tp_profit_usd,
                    'avg_tp_profit_usd': metrics.avg_tp_profit_usd,
                    'premature_exits': metrics.premature_exits,
                    'missed_opportunities_usd': metrics.missed_opportunities_usd,
                    'last_updated': metrics.last_updated.isoformat()
                }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"[TP Tracker] Failed to save metrics: {e}")
    
    def _load_metrics(self):
        """Load metrics from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for key, item in data.items():
                strat = item['strategy_id']
                sym = item['symbol']
                
                metrics = TPMetrics(
                    strategy_id=strat,
                    symbol=sym,
                    tp_attempts=item['tp_attempts'],
                    tp_hits=item['tp_hits'],
                    tp_misses=item['tp_misses'],
                    tp_hit_rate=item['tp_hit_rate'],
                    total_slippage_pct=item['total_slippage_pct'],
                    avg_slippage_pct=item['avg_slippage_pct'],
                    max_slippage_pct=item['max_slippage_pct'],
                    avg_time_to_tp_minutes=item['avg_time_to_tp_minutes'],
                    fastest_tp_minutes=item['fastest_tp_minutes'],
                    slowest_tp_minutes=item['slowest_tp_minutes'],
                    total_tp_profit_usd=item['total_tp_profit_usd'],
                    avg_tp_profit_usd=item['avg_tp_profit_usd'],
                    premature_exits=item['premature_exits'],
                    missed_opportunities_usd=item['missed_opportunities_usd'],
                    last_updated=datetime.fromisoformat(item['last_updated'])
                )
                
                self.metrics[(strat, sym)] = metrics
                
        except Exception as e:
            self.logger.warning(f"[TP Tracker] Failed to load metrics: {e}")


# Global singleton
_tp_performance_tracker: Optional[TPPerformanceTracker] = None


def get_tp_performance_tracker() -> TPPerformanceTracker:
    """Get or create TP Performance Tracker singleton."""
    global _tp_performance_tracker
    if _tp_performance_tracker is None:
        _tp_performance_tracker = TPPerformanceTracker()
    return _tp_performance_tracker


# Alias for convenience
def get_tp_tracker() -> TPPerformanceTracker:
    """Alias for get_tp_performance_tracker()."""
    return get_tp_performance_tracker()

"""
Exit Brain v3 Metrics - Performance tracking and feedback.

Integrates with tp_performance_tracker to improve exit strategies over time.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExitMetrics:
    """Track and analyze exit performance for continuous improvement"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.logger = logging.getLogger(__name__)
        self.tp_performance = None
        
        # Try to import TP tracker
        try:
            from backend.services.monitoring.tp_performance_tracker import get_tp_performance_tracker
            self.tp_performance = get_tp_performance_tracker()
            self.logger.info("[OK] TP Performance Tracker integrated")
        except ImportError as e:
            self.logger.warning(f"[WARNING] TP tracker unavailable: {e}")
    
    async def record_exit_leg_hit(
        self,
        symbol: str,
        leg_kind: str,
        realized_pct: float,
        time_to_exit_seconds: int
    ):
        """Record when an exit leg is hit"""
        if self.tp_performance:
            # Forward to TP tracker
            await self.tp_performance.record_tp_hit(
                symbol=symbol,
                tp_level=leg_kind,
                realized_pct=realized_pct,
                time_to_hit=time_to_exit_seconds
            )
            
            self.logger.debug(
                f"[EXIT METRICS] {symbol} {leg_kind} hit: "
                f"{realized_pct:.2f}% in {time_to_exit_seconds}s"
            )
    
    def get_recommended_adjustments(self, symbol: str) -> Optional[Dict]:
        """
        Get suggestions for exit strategy adjustments based on performance.
        
        Returns:
            {
                "tp_adjustment": str,  # "TIGHTEN", "WIDEN", "OK"
                "sl_adjustment": str,  # "TIGHTEN", "WIDEN", "OK"
                "reason": str
            }
        """
        if not self.tp_performance:
            return None
        
        # Analyze TP hit rates
        stats = self.tp_performance.get_stats(symbol)
        if not stats:
            return None
        
        tp1_rate = stats.get("tp1_hit_rate", 0.0)
        tp2_rate = stats.get("tp2_hit_rate", 0.0)
        tp3_rate = stats.get("tp3_hit_rate", 0.0)
        
        # Decision logic
        if tp1_rate < 0.30:  # TP1 hit < 30%
            return {
                "tp_adjustment": "TIGHTEN",
                "sl_adjustment": "OK",
                "reason": f"TP1 too far ({tp1_rate:.0%} hit rate)"
            }
        elif tp3_rate < 0.10:  # TP3 almost never hit
            return {
                "tp_adjustment": "TIGHTEN",
                "sl_adjustment": "OK",
                "reason": f"TP3 unrealistic ({tp3_rate:.0%} hit rate)"
            }
        elif tp1_rate > 0.80:  # TP1 hit too often
            return {
                "tp_adjustment": "WIDEN",
                "sl_adjustment": "OK",
                "reason": f"TP1 too tight ({tp1_rate:.0%} hit rate)"
            }
        else:
            return {
                "tp_adjustment": "OK",
                "sl_adjustment": "OK",
                "reason": "Exit levels performing well"
            }

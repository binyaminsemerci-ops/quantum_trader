"""
Position Monitor Enhancement - Dynamic Trailing TP Rearm
========================================================

Adds intelligent trailing stop tightening as positions become profitable.
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class DynamicTrailingManager:
    """
    Manages dynamic trailing stop adjustments based on position profitability.
    
    Features:
    - Tightens callback % as profit increases
    - Prevents premature exits with intelligent thresholds
    - Integrates with existing Position Monitor
    """
    
    def __init__(self):
        """Initialize Dynamic Trailing Manager."""
        self.logger = logging.getLogger(__name__)
        
        # Profit-based callback scaling
        self.profit_thresholds = [
            (0.02, 1.0),   # 2% profit → keep original callback
            (0.05, 0.75),  # 5% profit → 75% of original
            (0.10, 0.50),  # 10% profit → 50% of original
            (0.15, 0.30),  # 15% profit → 30% of original
            (0.20, 0.20),  # 20% profit → 20% of original
        ]
        
        self.min_callback_pct = 0.005  # Minimum 0.5% callback
        self.max_callback_pct = 0.05   # Maximum 5% callback
        
        # Track last adjustment times to prevent excessive updates
        self._last_adjustment_times: Dict[str, float] = {}
        self.min_adjustment_interval_seconds = 30
        
        self.logger.info("[Dynamic Trailing] Manager initialized")
    
    def calculate_optimal_callback(
        self,
        unrealized_pnl_pct: float,
        current_callback_pct: float,
        position_age_minutes: int
    ) -> Optional[float]:
        """
        Calculate optimal trailing stop callback based on profit level.
        
        Args:
            unrealized_pnl_pct: Current unrealized PnL percentage
            current_callback_pct: Current trailing callback percentage
            position_age_minutes: How long position has been open
            
        Returns:
            New optimal callback % or None if no change needed
        """
        # Only tighten for profitable positions
        if unrealized_pnl_pct < 0.02:
            return None
        
        # Don't adjust very young positions (let them breathe)
        if position_age_minutes < 5:
            return None
        
        # Find appropriate scaling factor based on profit level
        scale_factor = 1.0
        for profit_threshold, scale in self.profit_thresholds:
            if unrealized_pnl_pct >= profit_threshold:
                scale_factor = scale
        
        # Calculate new callback
        new_callback = current_callback_pct * scale_factor
        new_callback = max(self.min_callback_pct, min(self.max_callback_pct, new_callback))
        
        # Only return if significantly different (> 10% change)
        if abs(new_callback - current_callback_pct) / current_callback_pct > 0.1:
            return new_callback
        
        return None
    
    def should_update_trailing_stop(
        self,
        symbol: str,
        current_time: float,
        unrealized_pnl_pct: float
    ) -> bool:
        """
        Check if trailing stop should be updated for this symbol.
        
        Args:
            symbol: Trading symbol
            current_time: Current timestamp
            unrealized_pnl_pct: Current unrealized PnL percentage
            
        Returns:
            True if update is warranted
        """
        # Only update profitable positions
        if unrealized_pnl_pct < 0.02:
            return False
        
        # Check rate limiting
        last_update = self._last_adjustment_times.get(symbol, 0)
        if current_time - last_update < self.min_adjustment_interval_seconds:
            return False
        
        return True
    
    def record_adjustment(self, symbol: str, current_time: float):
        """Record that an adjustment was made for rate limiting."""
        self._last_adjustment_times[symbol] = current_time
    
    def get_partial_tp_levels(
        self,
        entry_price: float,
        tp_price: float,
        side: str,
        unrealized_pnl_pct: float
    ) -> list:
        """
        Calculate partial TP levels based on profit.
        
        Args:
            entry_price: Position entry price
            tp_price: Full TP price
            side: Position side (LONG/SHORT)
            unrealized_pnl_pct: Current unrealized PnL
            
        Returns:
            List of (price, qty_pct, reason) tuples
        """
        levels = []
        
        # For highly profitable positions, add multiple TP levels
        if unrealized_pnl_pct > 0.10:
            # 3-level TP: 30% at 50% profit, 40% at 75% profit, 30% at full TP
            if side.upper() in ['LONG', 'BUY']:
                mid_price_50 = entry_price + (tp_price - entry_price) * 0.5
                mid_price_75 = entry_price + (tp_price - entry_price) * 0.75
            else:
                mid_price_50 = entry_price - (entry_price - tp_price) * 0.5
                mid_price_75 = entry_price - (entry_price - tp_price) * 0.75
            
            levels = [
                (mid_price_50, 0.30, "Partial TP 50%"),
                (mid_price_75, 0.40, "Partial TP 75%"),
                (tp_price, 0.30, "Final TP 100%")
            ]
        elif unrealized_pnl_pct > 0.05:
            # 2-level TP: 50% at 66% profit, 50% at full TP
            if side.upper() in ['LONG', 'BUY']:
                mid_price = entry_price + (tp_price - entry_price) * 0.66
            else:
                mid_price = entry_price - (entry_price - tp_price) * 0.66
            
            levels = [
                (mid_price, 0.50, "Partial TP 66%"),
                (tp_price, 0.50, "Final TP 100%")
            ]
        else:
            # Single full TP
            levels = [(tp_price, 1.0, "Full TP")]
        
        return levels


# Global singleton
_dynamic_trailing_manager: Optional[DynamicTrailingManager] = None


def get_dynamic_trailing_manager() -> DynamicTrailingManager:
    """Get or create Dynamic Trailing Manager singleton."""
    global _dynamic_trailing_manager
    if _dynamic_trailing_manager is None:
        _dynamic_trailing_manager = DynamicTrailingManager()
    return _dynamic_trailing_manager

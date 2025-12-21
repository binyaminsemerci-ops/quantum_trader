"""
ExitBrain v3.5 Integration Bridge
Connects adaptive leverage engine to ExitBrain v3 adapter
"""
import os
import sys
from typing import Dict, Optional
import logging

# Import from microservices package (works in Docker)
try:
    from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
    from microservices.exitbrain_v3_5.pnl_tracker import PnLTracker
except ImportError:
    # Fallback for local development
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'microservices'))
    from exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
    from exitbrain_v3_5.pnl_tracker import PnLTracker

logger = logging.getLogger(__name__)


class ExitBrainV35Integration:
    """
    Integration bridge for ExitBrain v3.5 features
    
    Provides leverage-aware TP/SL adjustment to ExitBrain v3 adapter
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize ExitBrain v3.5 integration
        
        Args:
            enabled: Whether adaptive leverage features are enabled
        """
        self.enabled = enabled
        
        if self.enabled:
            self.adaptive_engine = AdaptiveLeverageEngine()
            self.pnl_tracker = PnLTracker(max_history=20)
            logger.info("✅ ExitBrain v3.5 Adaptive Leverage Engine initialized")
        else:
            self.adaptive_engine = None
            self.pnl_tracker = None
            logger.info("⚠️ ExitBrain v3.5 disabled (using v3 defaults)")
    
    def compute_adaptive_levels(
        self,
        leverage: float,
        volatility_factor: float = 1.0,
        confidence: float = 0.5,
        base_tp: Optional[float] = None,
        base_sl: Optional[float] = None
    ) -> Dict:
        """
        Compute adaptive TP/SL levels
        
        Args:
            leverage: Position leverage (1-100x)
            volatility_factor: Cross-exchange volatility multiplier
            confidence: AI signal confidence (0-1)
            base_tp: Override base TP (optional)
            base_sl: Override base SL (optional)
            
        Returns:
            Dictionary with tp1, tp2, tp3, sl, LSF, harvest_scheme, adjustment
        """
        if not self.enabled:
            # Return v3 defaults if disabled
            return {
                'tp1': 0.01,
                'tp2': 0.015,
                'tp3': 0.02,
                'sl': 0.005,
                'LSF': 1.0,
                'harvest_scheme': [0.33, 0.33, 0.34],
                'adjustment': 1.0
            }
        
        # Override base levels if provided
        if base_tp:
            self.adaptive_engine.base_tp = base_tp
        if base_sl:
            self.adaptive_engine.base_sl = base_sl
        
        # Compute levels
        levels = self.adaptive_engine.compute_levels(leverage, volatility_factor)
        
        # Get harvest scheme
        harvest_scheme = self.adaptive_engine.get_harvest_scheme(leverage)
        
        # Get PnL-based adjustment
        avg_pnl = self.pnl_tracker.avg_last_20() if self.pnl_tracker else 0.0
        adjustment = self.adaptive_engine.optimize_based_on_pnl(avg_pnl, confidence)
        
        # Apply adjustment to levels
        for key in ['tp1', 'tp2', 'tp3', 'sl']:
            levels[key] *= adjustment
        
        # Validate levels
        is_valid = self.adaptive_engine.validate_levels(levels, leverage)
        
        if not is_valid:
            logger.error(f"❌ Invalid levels computed for {leverage}x, using defaults")
            return {
                'tp1': 0.01,
                'tp2': 0.015,
                'tp3': 0.02,
                'sl': 0.005,
                'LSF': 1.0,
                'harvest_scheme': [0.33, 0.33, 0.34],
                'adjustment': 1.0
            }
        
        # Add additional fields
        levels['harvest_scheme'] = harvest_scheme
        levels['adjustment'] = adjustment
        levels['avg_pnl_last_20'] = avg_pnl
        
        logger.info(
            f"Adaptive levels for {leverage}x (volatility={volatility_factor:.2f}): "
            f"TP1={levels['tp1']:.3%}, TP2={levels['tp2']:.3%}, TP3={levels['tp3']:.3%}, "
            f"SL={levels['sl']:.3%}, Adjustment={adjustment:.2f}"
        )
        
        return levels
    
    def record_trade_result(
        self,
        symbol: str,
        leverage: float,
        pnl: float
    ):
        """
        Record trade result for PnL-based optimization
        
        Args:
            symbol: Trading pair
            leverage: Leverage used
            pnl: Trade PnL as percentage (e.g., 0.05 = +5%)
        """
        if not self.enabled or not self.pnl_tracker:
            return
        
        self.pnl_tracker.add_trade(pnl, symbol, leverage)
        
        # Log PnL stats
        stats = self.pnl_tracker.get_stats()
        logger.info(
            f"Trade recorded: {symbol} {leverage}x → PnL {pnl:+.2%} | "
            f"Rolling avg: {stats['avg_pnl']:+.2%}, Win rate: {stats['win_rate']:.1%}"
        )
    
    def get_pnl_stats(self) -> Dict:
        """Get PnL tracker statistics"""
        if not self.enabled or not self.pnl_tracker:
            return {
                'avg_pnl': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        return self.pnl_tracker.get_stats()


# Global singleton instance
_v35_integration = None


def get_v35_integration(enabled: Optional[bool] = None) -> ExitBrainV35Integration:
    """
    Get global ExitBrain v3.5 integration instance
    
    Args:
        enabled: Override enabled state (None = use env var)
        
    Returns:
        ExitBrainV35Integration singleton
    """
    global _v35_integration
    
    if _v35_integration is None:
        if enabled is None:
            enabled = os.getenv("ADAPTIVE_LEVERAGE_ENABLED", "true").lower() == "true"
        
        _v35_integration = ExitBrainV35Integration(enabled=enabled)
    
    return _v35_integration

"""
ExitBrain v3.5 Integration Layer
Connects TradeIntentSubscriber to AdaptiveLeverageEngine
"""
import logging
from typing import Dict, Optional
from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
from microservices.exitbrain_v3_5.pnl_tracker import PnLTracker

logger = logging.getLogger(__name__)


class ExitBrainV35Integration:
    """Integration layer for ExitBrain v3.5 Adaptive Leverage"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        
        if self.enabled:
            try:
                # Initialize core engine with default base levels
                self.adaptive_engine = AdaptiveLeverageEngine(
                    base_tp=1.0,  # 1% base TP
                    base_sl=0.5   # 0.5% base SL
                )
                
                # Initialize PnL tracker
                self.pnl_tracker = PnLTracker(max_history=20)
                
                logger.info("✅ ExitBrain v3.5 initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ExitBrain v3.5: {e}")
                self.enabled = False
                raise
        else:
            logger.info("⏸️  ExitBrain v3.5 disabled")
    
    def compute_adaptive_levels(
        self,
        symbol: str,
        leverage: float,
        volatility_factor: float = 1.0,
        base_tp: Optional[float] = None,
        base_sl: Optional[float] = None,
        confidence: float = 0.8
    ) -> Dict:
        """
        Compute adaptive TP/SL levels based on leverage and volatility
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            leverage: Current leverage (e.g., 10.0)
            volatility_factor: Volatility multiplier (default 1.0)
            base_tp: Override base TP percentage (optional)
            base_sl: Override base SL percentage (optional)
            confidence: Confidence level for adjustments (0.0-1.0)
        
        Returns:
            Dict with keys: tp1, tp2, tp3, sl, LSF, harvest_scheme, avg_pnl_last_20
        """
        if not self.enabled:
            logger.warning("ExitBrain v3.5 disabled, returning default levels")
            return {
                'tp1': 0.01,
                'tp2': 0.015,
                'tp3': 0.02,
                'sl': 0.005,
                'LSF': 1.0,
                'harvest_scheme': [0.33, 0.33, 0.34],
                'avg_pnl_last_20': 0.0
            }
        
        try:
            logger.info(f"Computing adaptive levels for {symbol} at {leverage}x leverage")
            
            # Update base levels if provided
            if base_tp:
                self.adaptive_engine.base_tp = base_tp
            if base_sl:
                self.adaptive_engine.base_sl = base_sl
            
            # Compute levels (returns AdaptiveLevels dataclass)
            levels = self.adaptive_engine.compute_levels(
                self.adaptive_engine.base_tp, 
                self.adaptive_engine.base_sl, 
                leverage, 
                volatility_factor
            )
            
            # Get PnL stats for logging
            avg_pnl = self.pnl_tracker.avg_last_20() if self.pnl_tracker else 0.0
            
            # Convert AdaptiveLevels dataclass to dict for EventBus
            result = {
                'tp1': levels.tp1_pct,
                'tp2': levels.tp2_pct,
                'tp3': levels.tp3_pct,
                'sl': levels.sl_pct,
                'LSF': levels.lsf,
                'harvest_scheme': levels.harvest_scheme,
                'avg_pnl_last_20': avg_pnl
            }
            
            logger.info(
                f"✅ Adaptive levels for {leverage}x (volatility={volatility_factor:.2f}): "
                f"TP1={result['tp1']:.3f}%, TP2={result['tp2']:.3f}%, TP3={result['tp3']:.3f}%, "
                f"SL={result['sl']:.3f}%, LSF={result['LSF']:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error computing adaptive levels: {e}", exc_info=True)
            # Return safe defaults
            return {
                'tp1': 0.01,
                'tp2': 0.015,
                'tp3': 0.02,
                'sl': 0.005,
                'LSF': 1.0,
                'harvest_scheme': [0.33, 0.33, 0.34],
                'avg_pnl_last_20': 0.0
            }
    
    def record_trade_result(
        self,
        symbol: str,
        pnl_pct: float,
        leverage: float
    ) -> None:
        """Record trade result for adaptive learning"""
        if not self.enabled:
            return
        
        try:
            self.pnl_tracker.add_result(pnl_pct)
            logger.info(f"📊 Recorded trade result: {symbol} @ {leverage}x = {pnl_pct:.2%} PnL")
        except Exception as e:
            logger.error(f"❌ Error recording trade result: {e}")

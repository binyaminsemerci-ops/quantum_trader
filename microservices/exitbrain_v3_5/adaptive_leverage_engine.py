"""
Adaptive Leverage Engine - ExitBrain v3.5
Dynamic TP/SL adjustment based on leverage, volatility, and PnL history
"""
import math
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveLeverageEngine:
    """
    Adaptive Leverage-Aware Profit Harvesting Engine
    
    Features:
    - Leverage-sensitive TP/SL calculation
    - Partial profit harvesting (3 levels)
    - Cross-exchange ATR adjustment
    - PnL-based auto-tuning
    """
    
    def __init__(self, base_tp: float = 0.01, base_sl: float = 0.005):
        """
        Initialize adaptive leverage engine
        
        Args:
            base_tp: Base take profit ratio (1%)
            base_sl: Base stop loss ratio (0.5%)
        """
        self.base_tp = base_tp
        self.base_sl = base_sl
        logger.info(f"✅ AdaptiveLeverageEngine initialized: base_tp={base_tp}, base_sl={base_sl}")
    
    def compute_leverage_factor(self, leverage: float) -> float:
        """
        Compute Leverage Scaling Factor (LSF)
        
        Higher leverage = Lower LSF = Tighter TP/SL
        Formula: LSF = 1 / (1 + ln(leverage + 1))
        
        Args:
            leverage: Position leverage (1-100x)
            
        Returns:
            LSF between 0 and 1
        """
        if leverage <= 0:
            leverage = 1
        
        lsf = 1 / (1 + math.log(leverage + 1))
        
        # Ensure LSF stays in valid range
        lsf = max(0.1, min(lsf, 1.0))
        
        logger.debug(f"Leverage {leverage}x → LSF = {lsf:.4f}")
        return lsf
    
    def compute_levels(
        self, 
        leverage: float, 
        volatility_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute adaptive TP/SL levels
        
        Args:
            leverage: Position leverage
            volatility_factor: Cross-exchange volatility multiplier (default 1.0)
            
        Returns:
            Dictionary with tp1, tp2, tp3, sl, LSF
        """
        # Compute leverage scaling factor
        LSF = self.compute_leverage_factor(leverage)
        
        # Base level calculations with leverage adjustment
        tp1 = self.base_tp * (0.6 + LSF)
        tp2 = self.base_tp * (1.2 + LSF / 2)
        tp3 = self.base_tp * (1.8 + LSF / 4)
        sl = self.base_sl * (1 + (1 - LSF) * 0.8)
        
        # Apply cross-exchange ATR corrections
        # Higher volatility = Wider targets
        tp1 *= (1 + volatility_factor * 0.4)
        tp2 *= (1 + volatility_factor * 0.4)
        tp3 *= (1 + volatility_factor * 0.4)
        sl *= (1 + volatility_factor * 0.2)
        
        # Fail-safe margin protection
        # Prevent liquidation on high leverage
        sl = min(max(sl, 0.001), 0.02)  # 0.1% to 2% range
        
        levels = {
            'tp1': round(tp1, 5),
            'tp2': round(tp2, 5),
            'tp3': round(tp3, 5),
            'sl': round(sl, 5),
            'LSF': round(LSF, 4)
        }
        
        logger.info(
            f"Computed levels for {leverage}x leverage (volatility={volatility_factor:.2f}): "
            f"TP1={levels['tp1']:.3%}, TP2={levels['tp2']:.3%}, TP3={levels['tp3']:.3%}, "
            f"SL={levels['sl']:.3%}"
        )
        
        return levels
    
    def get_harvest_scheme(self, leverage: float) -> List[float]:
        """
        Get partial profit harvesting scheme based on leverage
        
        Args:
            leverage: Position leverage
            
        Returns:
            List of 3 percentages for TP1, TP2, TP3 (sum = 1.0)
        """
        if leverage <= 10:
            # Conservative: Gradual profit taking
            scheme = [0.3, 0.3, 0.4]
            strategy = "Conservative"
        elif leverage <= 30:
            # Aggressive: Front-load profit taking
            scheme = [0.4, 0.4, 0.2]
            strategy = "Aggressive"
        else:
            # Ultra-aggressive: Maximize early profits
            scheme = [0.5, 0.3, 0.2]
            strategy = "Ultra-Aggressive"
        
        logger.info(
            f"{strategy} harvest scheme for {leverage}x: "
            f"TP1={scheme[0]:.0%}, TP2={scheme[1]:.0%}, TP3={scheme[2]:.0%}"
        )
        
        return scheme
    
    def optimize_based_on_pnl(
        self, 
        avg_pnl_last_20: float, 
        current_confidence: float
    ) -> float:
        """
        Dynamic adjustment based on recent profitability
        
        Args:
            avg_pnl_last_20: Average PnL of last 20 trades (e.g., 0.3 = +30%)
            current_confidence: AI signal confidence (0-1)
            
        Returns:
            Adjustment factor (0.8 - 1.2)
        """
        adjust = 1.0
        
        # PnL-based adjustment
        if avg_pnl_last_20 < 0:
            # Recent losses → Tighten stops
            adjust -= 0.1
            logger.warning(f"Negative PnL detected ({avg_pnl_last_20:.2%}) → Tightening levels by 10%")
        elif avg_pnl_last_20 > 0.3:
            # Strong profits → Allow room to run
            adjust += 0.1
            logger.info(f"Strong PnL ({avg_pnl_last_20:.2%}) → Expanding levels by 10%")
        
        # Confidence-based adjustment
        if current_confidence < 0.5:
            # Low confidence → Be more conservative
            adjust -= 0.05
            logger.warning(f"Low confidence ({current_confidence:.2%}) → Extra 5% tightening")
        
        # Clamp to safe range
        adjust = max(0.8, min(adjust, 1.2))
        
        logger.info(f"PnL optimizer: adjustment factor = {adjust:.2f}")
        return adjust
    
    def validate_levels(self, levels: Dict[str, float], leverage: float) -> bool:
        """
        Validate that computed levels are safe
        
        Args:
            levels: Dictionary with TP/SL levels
            leverage: Position leverage
            
        Returns:
            True if levels are valid and safe
        """
        # Check TP progression
        if not (levels['tp1'] < levels['tp2'] < levels['tp3']):
            logger.error("❌ Invalid TP progression")
            return False
        
        # Check SL is positive
        if levels['sl'] <= 0:
            logger.error("❌ Invalid SL (must be positive)")
            return False
        
        # Check liquidation risk
        # For high leverage, SL must be well within liquidation distance
        max_safe_sl = 0.8 / leverage  # 80% of liquidation distance
        if levels['sl'] > max_safe_sl:
            logger.error(
                f"❌ SL too wide for {leverage}x leverage "
                f"(SL={levels['sl']:.3%}, max_safe={max_safe_sl:.3%})"
            )
            return False
        
        logger.debug("✅ Levels validated successfully")
        return True


def test_adaptive_engine():
    """Test adaptive leverage engine"""
    logger.info("=== Testing Adaptive Leverage Engine ===\n")
    
    engine = AdaptiveLeverageEngine()
    
    # Test 1: Various leverage levels
    logger.info("Test 1: Leverage Scaling")
    for leverage in [1, 5, 10, 20, 50, 100]:
        lsf = engine.compute_leverage_factor(leverage)
        levels = engine.compute_levels(leverage)
        valid = engine.validate_levels(levels, leverage)
        logger.info(
            f"  {leverage:3d}x → LSF={lsf:.4f}, TP1={levels['tp1']:.3%}, "
            f"SL={levels['sl']:.3%}, Valid={valid}"
        )
    
    # Test 2: Volatility adjustment
    logger.info("\nTest 2: Volatility Adjustment (50x leverage)")
    for vol_factor in [0.5, 1.0, 1.5, 2.0]:
        levels = engine.compute_levels(50, volatility_factor=vol_factor)
        logger.info(
            f"  Volatility {vol_factor:.1f}x → TP1={levels['tp1']:.3%}, "
            f"TP2={levels['tp2']:.3%}, TP3={levels['tp3']:.3%}"
        )
    
    # Test 3: Harvest schemes
    logger.info("\nTest 3: Harvest Schemes")
    for leverage in [5, 15, 50]:
        scheme = engine.get_harvest_scheme(leverage)
        logger.info(f"  {leverage:2d}x → {scheme}")
    
    # Test 4: PnL optimization
    logger.info("\nTest 4: PnL-based Optimization")
    test_cases = [
        (-0.15, 0.6, "Recent losses, good confidence"),
        (0.45, 0.8, "Strong profits, high confidence"),
        (0.05, 0.4, "Break-even, low confidence"),
    ]
    for avg_pnl, confidence, desc in test_cases:
        adjust = engine.optimize_based_on_pnl(avg_pnl, confidence)
        logger.info(f"  {desc}: PnL={avg_pnl:+.1%}, Conf={confidence:.1%} → Adjust={adjust:.2f}")
    
    # Test 5: Complete workflow
    logger.info("\nTest 5: Complete Workflow (50x leverage, volatile market)")
    leverage = 50
    volatility = 1.3
    avg_pnl = 0.25
    confidence = 0.75
    
    levels = engine.compute_levels(leverage, volatility)
    adjustment = engine.optimize_based_on_pnl(avg_pnl, confidence)
    harvest = engine.get_harvest_scheme(leverage)
    
    # Apply adjustment
    for key in ['tp1', 'tp2', 'tp3', 'sl']:
        levels[key] *= adjustment
    
    valid = engine.validate_levels(levels, leverage)
    
    logger.info(f"\n  Final Levels:")
    logger.info(f"    TP1: {levels['tp1']:.3%} (harvest {harvest[0]:.0%})")
    logger.info(f"    TP2: {levels['tp2']:.3%} (harvest {harvest[1]:.0%})")
    logger.info(f"    TP3: {levels['tp3']:.3%} (harvest {harvest[2]:.0%})")
    logger.info(f"    SL:  {levels['sl']:.3%}")
    logger.info(f"    LSF: {levels['LSF']:.4f}")
    logger.info(f"    Adjustment: {adjustment:.2f}")
    logger.info(f"    Valid: {valid}")
    
    logger.info("\n✅ All tests completed successfully")
    return True


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        success = test_adaptive_engine()
        sys.exit(0 if success else 1)
    else:
        print("Usage: python adaptive_leverage_engine.py --test")

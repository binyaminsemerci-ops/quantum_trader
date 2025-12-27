"""
Adaptive Leverage Engine - ExitBrain v3.5
Leverage-aware TP/SL + multi-stage harvesting with fail-safe clamps

Implements:
- Leverage Sensitivity Factor (LSF): LSF = 1 / (1 + log(leverage + 1))
- Multi-stage TP levels with harvest schemes
- Fail-safe clamps: SL [0.1%, 2.0%], TP min 0.3%, SL min 0.15%
- Dynamic adjustment based on volatility, funding, divergence
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveLevels:
    """Result of adaptive level calculation"""
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float
    sl_pct: float
    harvest_scheme: List[float]
    lsf: float


class AdaptiveLeverageEngine:
    """
    Adaptive Leverage-Aware TP/SL engine (ExitBrain v3 compatible).

    Inputs are percentages expressed as decimals:
      - 1%  = 0.01
      - 0.5% = 0.005
    
    Features:
    - LSF-based TP/SL calculation
    - Multi-stage profit harvesting
    - Fail-safe clamps for safety
    - Volatility, funding, divergence adjustments
    """

    # Hard safety clamps (from design spec)
    SL_CLAMP_MIN = 0.001   # 0.1%
    SL_CLAMP_MAX = 0.02    # 2.0%

    # Soft minimums (from design spec)
    TP_MIN = 0.003         # 0.3%
    SL_MIN = 0.0015        # 0.15%

    def __init__(self, base_tp: float = 0.01, base_sl: float = 0.005):
        """
        Initialize adaptive leverage engine
        
        Args:
            base_tp: Base take profit ratio (default 1%)
            base_sl: Base stop loss ratio (default 0.5%)
        """
        self.base_tp = base_tp
        self.base_sl = base_sl
        logger.info(
            f"[AdaptiveLeverageEngine] Initialized | "
            f"Base TP: {base_tp*100:.2f}% | Base SL: {base_sl*100:.2f}%"
        )
    
    def compute_lsf(self, leverage: float) -> float:
        """
        Compute Leverage Sensitivity Factor (LSF)
        
        Formula: LSF = 1 / (1 + ln(leverage + 1))
        Higher leverage → Lower LSF → Tighter TP/SL
        
        Args:
            leverage: Position leverage (1-100x)
            
        Returns:
            LSF between 0 and 1
        """
        lev = max(float(leverage), 1.0)
        lsf = 1.0 / (1.0 + math.log(lev + 1.0))
        return lsf
        lsf = 1.0 / (1.0 + math.log(lev + 1.0))
        return lsf
    
    def harvest_scheme_for(self, leverage: float) -> List[float]:
        """
        Get partial profit harvesting scheme based on leverage
        
        Args:
            leverage: Position leverage
            
        Returns:
            List of 3 percentages for [TP1, TP2, TP3] (sum = 1.0)
        """
        lev = max(float(leverage), 1.0)
        if lev <= 10:
            # Conservative: Gradual profit taking
            return [0.3, 0.3, 0.4]
        if lev <= 30:
            # Aggressive: Front-load profit taking
            return [0.4, 0.4, 0.2]
        # Ultra-aggressive: Maximize early profits
        return [0.5, 0.3, 0.2]
    
    def compute_levels(
        self,
        base_tp_pct: float,
        base_sl_pct: float,
        leverage: float,
        volatility_factor: Optional[float] = None,
        funding_delta: Optional[float] = None,
        exchange_divergence: Optional[float] = None,
    ) -> AdaptiveLevels:
        """
        Compute adaptive TP/SL levels with fail-safe clamps
        
        Args:
            base_tp_pct: Base take profit percentage (decimal)
            base_sl_pct: Base stop loss percentage (decimal)
            leverage: Position leverage
            volatility_factor: Normalized 0..1-ish (0 = calm). If None => 0
            funding_delta: Signed small value (e.g. -0.02..+0.02). If None => 0
            exchange_divergence: Normalized 0..1-ish. If None => 0
            
        Returns:
            AdaptiveLevels with all calculated parameters
        """
        lsf = self.compute_lsf(leverage)

        base_tp = max(float(base_tp_pct), 0.0)
        base_sl = max(float(base_sl_pct), 0.0)

        # Core LSF-based formula (from spec)
        tp1 = base_tp * (0.6 + lsf)
        tp2 = base_tp * (1.2 + lsf / 2.0)
        tp3 = base_tp * (1.8 + lsf / 4.0)
        sl = base_sl * (1.0 + (1.0 - lsf) * 0.8)

        # Optional adaptive modifiers (kept conservative)
        vf = float(volatility_factor) if volatility_factor is not None else 0.0
        fd = float(funding_delta) if funding_delta is not None else 0.0
        xd = float(exchange_divergence) if exchange_divergence is not None else 0.0

        # Funding can scale TP (more positive funding -> slightly higher TP targets)
        tp_scale = 1.0 + (fd * 0.8)
        # Divergence/volatility can widen SL a bit to avoid premature stop-outs
        sl_scale = 1.0 + (xd * 0.4) + (vf * 0.2)

        tp1 *= tp_scale
        tp2 *= tp_scale
        tp3 *= tp_scale
        sl *= sl_scale

        # Apply soft minimums
        tp1 = max(tp1, self.TP_MIN)
        tp2 = max(tp2, self.TP_MIN)
        tp3 = max(tp3, self.TP_MIN)
        sl = max(sl, self.SL_MIN)

        # Apply hard fail-safe clamps
        sl = min(max(sl, self.SL_CLAMP_MIN), self.SL_CLAMP_MAX)

        return AdaptiveLevels(
            tp1_pct=tp1,
            tp2_pct=tp2,
            tp3_pct=tp3,
            sl_pct=sl,
            harvest_scheme=self.harvest_scheme_for(leverage),
            lsf=lsf,
        )


def test_adaptive_engine():
    """Sanity tests for AdaptiveLeverageEngine"""
    logger.info("=== AdaptiveLeverageEngine Sanity Tests ===\n")
    
    engine = AdaptiveLeverageEngine(base_tp=0.01, base_sl=0.005)
    
    # Test 1: Low leverage should have higher TP1 than high leverage
    levels_low = engine.compute_levels(0.01, 0.005, 5)
    levels_high = engine.compute_levels(0.01, 0.005, 50)
    
    assert levels_low.tp1_pct > levels_high.tp1_pct, \
        f"Test 1 FAILED: Low leverage TP1 ({levels_low.tp1_pct:.4f}) should be > high leverage ({levels_high.tp1_pct:.4f})"
    logger.info(f"✅ Test 1 PASSED: 5x TP1={levels_low.tp1_pct:.4f} > 50x TP1={levels_high.tp1_pct:.4f}")
    
    # Test 2: High leverage should have wider SL than low leverage
    assert levels_high.sl_pct >= levels_low.sl_pct, \
        f"Test 2 FAILED: High leverage SL ({levels_high.sl_pct:.4f}) should be >= low leverage ({levels_low.sl_pct:.4f})"
    logger.info(f"✅ Test 2 PASSED: 50x SL={levels_high.sl_pct:.4f} >= 5x SL={levels_low.sl_pct:.4f}")
    
    # Test 3: Clamps work for extreme leverage
    levels_extreme = engine.compute_levels(0.01, 0.005, 100)
    assert engine.SL_CLAMP_MIN <= levels_extreme.sl_pct <= engine.SL_CLAMP_MAX, \
        f"Test 3 FAILED: SL {levels_extreme.sl_pct:.4f} outside clamps [{engine.SL_CLAMP_MIN}, {engine.SL_CLAMP_MAX}]"
    logger.info(f"✅ Test 3 PASSED: 100x SL={levels_extreme.sl_pct:.4f} within clamps [{engine.SL_CLAMP_MIN}, {engine.SL_CLAMP_MAX}]")
    
    # Test 4: Harvest schemes correct
    scheme_low = engine.harvest_scheme_for(5)
    scheme_mid = engine.harvest_scheme_for(20)
    scheme_high = engine.harvest_scheme_for(50)
    
    assert scheme_low == [0.3, 0.3, 0.4], f"Test 4a FAILED: Wrong scheme for 5x: {scheme_low}"
    assert scheme_mid == [0.4, 0.4, 0.2], f"Test 4b FAILED: Wrong scheme for 20x: {scheme_mid}"
    assert scheme_high == [0.5, 0.3, 0.2], f"Test 4c FAILED: Wrong scheme for 50x: {scheme_high}"
    logger.info(f"✅ Test 4 PASSED: Harvest schemes correct")
    
    # Test 5: TP minimums enforced
    levels_tiny = engine.compute_levels(0.001, 0.0001, 10)
    assert levels_tiny.tp1_pct >= engine.TP_MIN, \
        f"Test 5 FAILED: TP1 {levels_tiny.tp1_pct:.4f} < min {engine.TP_MIN}"
    logger.info(f"✅ Test 5 PASSED: TP min enforced ({levels_tiny.tp1_pct:.4f} >= {engine.TP_MIN})")
    
    logger.info("\n✅ All tests PASSED")
    return True


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        success = test_adaptive_engine()
        sys.exit(0 if success else 1)
    else:
        from .adaptive_leverage_engine import AdaptiveLeverageEngine
        print("Usage: python -m microservices.exitbrain_v3_5.adaptive_leverage_engine --test")
        print("\nQuick test:")
        engine = AdaptiveLeverageEngine()
        levels = engine.compute_levels(0.01, 0.005, 20)
        print(f"20x Leverage: TP1={levels.tp1_pct:.4f}, TP2={levels.tp2_pct:.4f}, TP3={levels.tp3_pct:.4f}, SL={levels.sl_pct:.4f}")
        print(f"Harvest scheme: {levels.harvest_scheme}")


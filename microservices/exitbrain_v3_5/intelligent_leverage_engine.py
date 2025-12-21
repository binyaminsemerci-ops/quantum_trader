"""
Intelligent Leverage Formula v2 (ILFv2) - Phase 4O+
Cross-Exchange Enabled Adaptive Leverage Calculation

Calculates optimal leverage (5-80x) based on:
- AI signal confidence
- Market volatility (ATR/stddev)
- Recent PnL trend
- Symbol-specific risk weights
- Margin utilization
- Cross-exchange price divergence (Phase 4M+)
- Funding rate imbalance

Formula:
    base = 5 + confidence × 75
    leverage = base × vol_factor × pnl_factor × symbol_factor × 
               margin_factor × divergence_factor × funding_factor
    
    Clamped to [5, 80]
"""

import numpy as np
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class LeverageCalculation:
    """Result of intelligent leverage calculation"""
    leverage: float  # Final calculated leverage (5-80)
    base_leverage: float  # Base before adjustments
    factors: Dict[str, float]  # Individual factor contributions
    reasoning: str  # Explanation of calculation
    confidence: float  # Input confidence
    clamped: bool  # Was leverage clamped to min/max?


class IntelligentLeverageEngine:
    """
    ILFv2 - Intelligent Leverage Formula v2
    
    Adaptive leverage calculation with cross-exchange awareness.
    Integrates with Phase 4M+ Cross-Exchange Intelligence.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Intelligent Leverage Engine
        
        Args:
            config: Configuration overrides
        """
        self.config = config or {}
        
        # Leverage bounds
        self.min_leverage = self.config.get("min_leverage", 5.0)
        self.max_leverage = self.config.get("max_leverage", 80.0)
        
        # Safety multipliers (can reduce leverage, never increase)
        self.safety_cap = self.config.get("safety_cap", 0.9)  # Max 90% of calculated
        
        # Factor weights (for tuning sensitivity)
        self.volatility_weight = self.config.get("volatility_weight", 1.5)
        self.pnl_weight = self.config.get("pnl_weight", 0.25)
        self.divergence_weight = self.config.get("divergence_weight", 1.0)
        self.funding_weight = self.config.get("funding_weight", 10.0)
        
        # Statistics tracking
        self.calculation_count = 0
        self.avg_leverage = 0.0
        self.leverage_history = []
        
        logger.info(
            f"[ILF-v2] Initialized | "
            f"Range: {self.min_leverage}-{self.max_leverage}x | "
            f"Safety Cap: {self.safety_cap*100:.0f}%"
        )
    
    def calculate_leverage(
        self,
        confidence: float,
        volatility: float,
        pnl_trend: float,
        symbol_risk: float = 1.0,
        margin_util: float = 0.0,
        exch_divergence: float = 0.0,
        funding_rate: float = 0.0
    ) -> LeverageCalculation:
        """
        Calculate optimal leverage using ILFv2 formula
        
        Args:
            confidence: AI signal confidence [0-1]
            volatility: Normalized volatility [0-3] (ATR/stddev)
            pnl_trend: Rolling PnL trend [-1 to +1]
            symbol_risk: Symbol risk weight [0.5-1.5] (default 1.0)
            margin_util: Used margin fraction [0-1]
            exch_divergence: Cross-exchange price divergence [0-1] (Phase 4M+)
            funding_rate: Funding rate bias [-0.05 to +0.05]
        
        Returns:
            LeverageCalculation with detailed breakdown
        """
        # Input validation
        confidence = max(0.0, min(1.0, confidence))
        volatility = max(0.0, min(3.0, volatility))
        pnl_trend = max(-1.0, min(1.0, pnl_trend))
        symbol_risk = max(0.5, min(1.5, symbol_risk))
        margin_util = max(0.0, min(1.0, margin_util))
        exch_divergence = max(0.0, min(1.0, exch_divergence))
        funding_rate = max(-0.05, min(0.05, funding_rate))
        
        # Step 1: Base leverage from confidence
        # Range: 5x (confidence=0) to 80x (confidence=1)
        base_leverage = self.min_leverage + (confidence * (self.max_leverage - self.min_leverage))
        
        # Step 2: Volatility factor
        # Lower leverage when volatile
        # Formula: max(0.2, 1.5 - volatility)
        vol_factor = max(0.2, self.volatility_weight - volatility)
        
        # Step 3: PnL trend factor
        # Reward consistent profit, penalize losses
        # Formula: 1 + (pnl_trend × 0.25)
        # Range: 0.75 (losing) to 1.25 (winning)
        pnl_factor = 1.0 + (pnl_trend * self.pnl_weight)
        
        # Step 4: Symbol risk factor
        # Reduce leverage for risky symbols
        # Formula: 1 / symbol_risk
        # Range: 0.67 (risky) to 2.0 (safe)
        symbol_factor = 1.0 / symbol_risk
        
        # Step 5: Margin utilization factor
        # Reduce leverage when margin nearly full
        # Formula: max(0.3, 1 - margin_util)
        # Range: 0.3 (90% used) to 1.0 (0% used)
        margin_factor = max(0.3, 1.0 - margin_util)
        
        # Step 6: Cross-exchange divergence factor (Phase 4M+)
        # Lower leverage if exchanges disagree on price
        # Formula: max(0.5, 1 - exch_divergence)
        # Range: 0.5 (100% divergence) to 1.0 (0% divergence)
        divergence_factor = max(0.5, 1.0 - (exch_divergence * self.divergence_weight))
        
        # Step 7: Funding rate factor
        # Penalize extreme funding (both positive and negative)
        # Formula: 1 - abs(funding_rate × 10)
        # Range: 0.5 (extreme) to 1.0 (neutral)
        funding_factor = max(0.5, 1.0 - abs(funding_rate * self.funding_weight))
        
        # Calculate final leverage
        leverage = (
            base_leverage * 
            vol_factor * 
            pnl_factor * 
            symbol_factor * 
            margin_factor * 
            divergence_factor * 
            funding_factor
        )
        
        # Apply safety cap
        leverage *= self.safety_cap
        
        # Clamp to bounds
        original_leverage = leverage
        leverage = max(self.min_leverage, min(self.max_leverage, leverage))
        clamped = (leverage != original_leverage)
        
        # Build factors dict
        factors = {
            "base": base_leverage,
            "volatility": vol_factor,
            "pnl_trend": pnl_factor,
            "symbol_risk": symbol_factor,
            "margin_util": margin_factor,
            "divergence": divergence_factor,
            "funding": funding_factor,
            "safety_cap": self.safety_cap
        }
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            confidence, volatility, pnl_trend, exch_divergence, 
            funding_rate, factors, clamped
        )
        
        # Update statistics
        self.calculation_count += 1
        self.leverage_history.append(leverage)
        if len(self.leverage_history) > 100:
            self.leverage_history.pop(0)
        self.avg_leverage = np.mean(self.leverage_history)
        
        result = LeverageCalculation(
            leverage=float(leverage),
            base_leverage=float(base_leverage),
            factors=factors,
            reasoning=reasoning,
            confidence=confidence,
            clamped=clamped
        )
        
        logger.info(
            f"[ILF-v2] Calculated: {leverage:.1f}x | "
            f"Base: {base_leverage:.1f}x | "
            f"Conf: {confidence:.2f} | "
            f"Vol: {volatility:.2f} | "
            f"Div: {exch_divergence:.3f} | "
            f"Fund: {funding_rate:.4f}"
        )
        
        return result
    
    def _generate_reasoning(
        self,
        confidence: float,
        volatility: float,
        pnl_trend: float,
        exch_divergence: float,
        funding_rate: float,
        factors: Dict[str, float],
        clamped: bool
    ) -> str:
        """Generate human-readable explanation of calculation"""
        parts = []
        
        # Confidence
        if confidence >= 0.85:
            parts.append(f"High confidence ({confidence:.0%})")
        elif confidence >= 0.65:
            parts.append(f"Moderate confidence ({confidence:.0%})")
        else:
            parts.append(f"Low confidence ({confidence:.0%})")
        
        # Volatility
        if volatility > 2.0:
            parts.append(f"high volatility (-{(1-factors['volatility'])*100:.0f}%)")
        elif volatility > 1.5:
            parts.append(f"elevated volatility (-{(1-factors['volatility'])*100:.0f}%)")
        else:
            parts.append(f"normal volatility")
        
        # PnL trend
        if pnl_trend > 0.3:
            parts.append(f"strong profit trend (+{(factors['pnl_trend']-1)*100:.0f}%)")
        elif pnl_trend < -0.3:
            parts.append(f"loss trend ({(factors['pnl_trend']-1)*100:.0f}%)")
        
        # Cross-exchange divergence
        if exch_divergence > 0.05:
            parts.append(f"exchange divergence (-{(1-factors['divergence'])*100:.0f}%)")
        
        # Funding rate
        if abs(funding_rate) > 0.01:
            parts.append(f"extreme funding (-{(1-factors['funding'])*100:.0f}%)")
        
        # Clamping
        if clamped:
            parts.append("clamped to bounds")
        
        return ", ".join(parts) if parts else "neutral conditions"
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return {
            "calculation_count": self.calculation_count,
            "avg_leverage": round(self.avg_leverage, 2),
            "current_range": f"{self.min_leverage}-{self.max_leverage}x",
            "safety_cap": f"{self.safety_cap*100:.0f}%",
            "recent_leverages": [round(x, 1) for x in self.leverage_history[-10:]]
        }


# Convenience function for direct use
def intelligent_leverage(
    confidence: float,
    volatility: float,
    pnl_trend: float,
    symbol_risk: float = 1.0,
    margin_util: float = 0.0,
    exch_divergence: float = 0.0,
    funding_rate: float = 0.0
) -> float:
    """
    Calculate intelligent leverage (convenience function)
    
    Returns:
        float: Optimal leverage between 5-80x
    """
    engine = IntelligentLeverageEngine()
    result = engine.calculate_leverage(
        confidence=confidence,
        volatility=volatility,
        pnl_trend=pnl_trend,
        symbol_risk=symbol_risk,
        margin_util=margin_util,
        exch_divergence=exch_divergence,
        funding_rate=funding_rate
    )
    return result.leverage


# Global singleton
_leverage_engine: Optional[IntelligentLeverageEngine] = None


def get_leverage_engine(config: Optional[Dict] = None) -> IntelligentLeverageEngine:
    """Get or create global IntelligentLeverageEngine instance"""
    global _leverage_engine
    
    if _leverage_engine is None:
        _leverage_engine = IntelligentLeverageEngine(config=config)
        logger.info("[ILF-v2] Global engine initialized")
    
    return _leverage_engine

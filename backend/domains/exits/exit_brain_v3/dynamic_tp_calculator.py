"""
Dynamic TP Calculator - AI-driven TP sizing based on position characteristics.

Takes into account:
- Position size (USD value)
- Leverage (higher leverage = tighter TPs)
- Coin volatility (volatile coins = wider TPs)
- Market volume/liquidity (can we actually exit?)
- Market regime (trending vs choppy)
- Entry confidence (from RL model)
- Current unrealized PnL (adjust on the fly)

Replaces fixed TP percentages with adaptive sizing.
"""
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DynamicTPResult:
    """Result from dynamic TP calculation"""
    tp_levels: List[Tuple[float, float]]  # [(tp_pct, size_fraction), ...]
    reasoning: str  # Explanation of why these levels were chosen
    confidence: float  # 0-1, how confident are we in these levels?
    risk_adjusted: bool  # Was this adjusted for high risk?


class DynamicTPCalculator:
    """
    Calculate optimal TP levels dynamically based on position characteristics.
    
    Philosophy:
    - Larger positions → Tighter TPs (harder to exit large size)
    - Higher leverage → Tighter TPs (more risk, take profit faster)
    - Volatile coins → Wider TPs (let volatility work for us)
    - Low liquidity → Tighter TPs (exit before slippage kills us)
    - Trending market → Wider TPs (let profits run)
    - Choppy market → Tighter TPs (take profit before reversal)
    - High confidence → Wider TPs (trust the signal)
    - Low confidence → Tighter TPs (defensive exits)
    """
    
    def __init__(self):
        """Initialize calculator with baseline parameters"""
        # Baseline TP levels (percentage from entry)
        # Format: [TP1_pct, TP2_pct, TP3_pct]
        self.baseline_tp_pcts = [0.015, 0.025, 0.040]  # 1.5%, 2.5%, 4.0%
        
        # Baseline size distribution
        # Format: [TP1_size, TP2_size, TP3_size] (must sum to 1.0)
        self.baseline_size_distribution = [0.30, 0.30, 0.40]  # 30%, 30%, 40%
        
        # Risk thresholds
        self.high_leverage_threshold = 15.0  # Above this = high risk
        self.large_position_usd_threshold = 2000.0  # Above this = large size
        self.high_volatility_threshold = 0.035  # Above this = volatile
        self.low_liquidity_threshold = 100000.0  # Below this USD 24h volume = low liquidity
        
        logger.info("[DYNAMIC_TP] Calculator initialized with adaptive sizing")
    
    def calculate_tp_levels(
        self,
        symbol: str,
        position_size_usd: float,
        leverage: float,
        volatility: float,
        market_regime: str,
        confidence: float,
        unrealized_pnl_pct: float,
        liquidity_24h_usd: Optional[float] = None
    ) -> DynamicTPResult:
        """
        Calculate optimal TP levels for position.
        
        Args:
            symbol: Trading pair (e.g., "XRPUSDT")
            position_size_usd: Position size in USD (notional value)
            leverage: Current leverage (e.g., 20.0)
            volatility: Recent volatility (e.g., 0.02 = 2% daily)
            market_regime: TRENDING, RANGE_BOUND, VOLATILE, NORMAL
            confidence: Entry confidence from RL (0.0-1.0)
            unrealized_pnl_pct: Current unrealized PnL %
            liquidity_24h_usd: Optional 24h trading volume in USD
            
        Returns:
            DynamicTPResult with optimal TP levels
        """
        logger.info(
            f"[DYNAMIC_TP] Calculating TPs for {symbol}: "
            f"size=${position_size_usd:.0f}, lev={leverage}x, "
            f"vol={volatility:.3f}, regime={market_regime}, conf={confidence:.2f}"
        )
        
        # Start with baseline
        tp_pcts = list(self.baseline_tp_pcts)
        size_dist = list(self.baseline_size_distribution)
        adjustments = []
        
        # Factor 1: Leverage adjustment
        if leverage >= self.high_leverage_threshold:
            # High leverage = tighter TPs (take profit faster, reduce risk)
            leverage_mult = 0.7  # 30% tighter
            tp_pcts = [tp * leverage_mult for tp in tp_pcts]
            adjustments.append(f"High leverage ({leverage}x) → -30% TP distance")
        elif leverage <= 5.0:
            # Low leverage = wider TPs (can afford to wait)
            leverage_mult = 1.3  # 30% wider
            tp_pcts = [tp * leverage_mult for tp in tp_pcts]
            adjustments.append(f"Low leverage ({leverage}x) → +30% TP distance")
        
        # Factor 2: Position size adjustment
        if position_size_usd >= self.large_position_usd_threshold:
            # Large position = tighter TPs + more levels (harder to exit)
            size_mult = 0.8  # 20% tighter
            tp_pcts = [tp * size_mult for tp in tp_pcts]
            # Shift distribution to exit more early
            size_dist = [0.40, 0.35, 0.25]  # Front-load exits
            adjustments.append(
                f"Large position (${position_size_usd:.0f}) → -20% TP distance, "
                f"front-load exits (40/35/25)"
            )
        
        # Factor 3: Volatility adjustment
        if volatility >= self.high_volatility_threshold:
            # High volatility = wider TPs (let big moves work for us)
            vol_mult = 1.4  # 40% wider
            tp_pcts = [tp * vol_mult for tp in tp_pcts]
            adjustments.append(f"High volatility ({volatility:.1%}) → +40% TP distance")
        elif volatility <= 0.015:
            # Low volatility = tighter TPs (small moves, take what we can)
            vol_mult = 0.75  # 25% tighter
            tp_pcts = [tp * vol_mult for tp in tp_pcts]
            adjustments.append(f"Low volatility ({volatility:.1%}) → -25% TP distance")
        
        # Factor 4: Market regime adjustment
        if market_regime == "TRENDING":
            # Trending = wider TPs (let trend work)
            tp_pcts = [tp * 1.5 for tp in tp_pcts]
            # Back-load exits (let runner run)
            size_dist = [0.20, 0.30, 0.50]
            adjustments.append("TRENDING regime → +50% TP distance, back-load (20/30/50)")
        elif market_regime == "VOLATILE" or market_regime == "RANGE_BOUND":
            # Choppy = tighter TPs (take profit before reversal)
            tp_pcts = [tp * 0.7 for tp in tp_pcts]
            adjustments.append(f"{market_regime} regime → -30% TP distance")
        
        # Factor 5: Confidence adjustment
        if confidence >= 0.85:
            # High confidence = wider TPs (trust the signal)
            tp_pcts = [tp * 1.3 for tp in tp_pcts]
            adjustments.append(f"High confidence ({confidence:.0%}) → +30% TP distance")
        elif confidence <= 0.6:
            # Low confidence = tighter TPs (defensive)
            tp_pcts = [tp * 0.75 for tp in tp_pcts]
            adjustments.append(f"Low confidence ({confidence:.0%}) → -25% TP distance")
        
        # Factor 6: Liquidity adjustment
        if liquidity_24h_usd and liquidity_24h_usd < self.low_liquidity_threshold:
            # Low liquidity = tighter TPs + front-load (exit before slippage)
            tp_pcts = [tp * 0.8 for tp in tp_pcts]
            size_dist = [0.45, 0.35, 0.20]
            adjustments.append(
                f"Low liquidity (${liquidity_24h_usd:.0f} 24h) → "
                f"-20% TP distance, front-load (45/35/20)"
            )
        
        # Factor 7: Unrealized PnL adjustment (dynamic rebalancing)
        if unrealized_pnl_pct >= 5.0:
            # Already in profit = tighten remaining TPs (lock profit)
            tp_pcts = [tp * 0.8 for tp in tp_pcts]
            adjustments.append(
                f"Already +{unrealized_pnl_pct:.1f}% PnL → "
                f"-20% remaining TP distance (lock profit)"
            )
        
        # Enforce minimum/maximum bounds
        tp_pcts = [
            max(0.005, min(0.10, tp))  # 0.5% - 10%
            for tp in tp_pcts
        ]
        
        # Build result
        tp_levels = list(zip(tp_pcts, size_dist))
        reasoning = "; ".join(adjustments) if adjustments else "Baseline TPs (no adjustments)"
        
        result = DynamicTPResult(
            tp_levels=tp_levels,
            reasoning=reasoning,
            confidence=confidence,
            risk_adjusted=(leverage >= self.high_leverage_threshold or 
                          position_size_usd >= self.large_position_usd_threshold)
        )
        
        logger.info(
            f"[DYNAMIC_TP] {symbol}: "
            f"TP1={tp_pcts[0]:.2%}({size_dist[0]:.0%}), "
            f"TP2={tp_pcts[1]:.2%}({size_dist[1]:.0%}), "
            f"TP3={tp_pcts[2]:.2%}({size_dist[2]:.0%})"
        )
        logger.info(f"[DYNAMIC_TP] Reasoning: {reasoning}")
        
        return result
    
    def adjust_tp_on_the_fly(
        self,
        current_tp_levels: List[Tuple[float, float]],
        new_unrealized_pnl_pct: float,
        new_regime: str
    ) -> Optional[DynamicTPResult]:
        """
        Dynamically adjust remaining TPs based on new market conditions.
        
        Called during position monitoring when conditions change significantly.
        
        Args:
            current_tp_levels: Current TP configuration
            new_unrealized_pnl_pct: Updated unrealized PnL %
            new_regime: Updated market regime
            
        Returns:
            New DynamicTPResult if adjustment needed, None if current levels OK
        """
        # If already in significant profit and regime turned choppy → tighten
        if new_unrealized_pnl_pct >= 8.0 and new_regime in ("VOLATILE", "RANGE_BOUND"):
            logger.warning(
                f"[DYNAMIC_TP] Live adjustment: +{new_unrealized_pnl_pct:.1f}% PnL "
                f"but regime={new_regime} → Tightening remaining TPs by 30%"
            )
            adjusted_levels = [
                (tp_pct * 0.7, size_frac)
                for tp_pct, size_frac in current_tp_levels
            ]
            return DynamicTPResult(
                tp_levels=adjusted_levels,
                reasoning=f"Live adjustment: {new_regime} regime with +{new_unrealized_pnl_pct:.1f}% profit",
                confidence=0.8,
                risk_adjusted=True
            )
        
        # No adjustment needed
        return None


# Global instance
_calculator = DynamicTPCalculator()


def calculate_dynamic_tp_levels(
    symbol: str,
    position_size_usd: float,
    leverage: float,
    volatility: float,
    market_regime: str,
    confidence: float,
    unrealized_pnl_pct: float = 0.0,
    liquidity_24h_usd: Optional[float] = None
) -> DynamicTPResult:
    """
    Convenience function for calculating dynamic TP levels.
    
    See DynamicTPCalculator.calculate_tp_levels for full documentation.
    """
    return _calculator.calculate_tp_levels(
        symbol=symbol,
        position_size_usd=position_size_usd,
        leverage=leverage,
        volatility=volatility,
        market_regime=market_regime,
        confidence=confidence,
        unrealized_pnl_pct=unrealized_pnl_pct,
        liquidity_24h_usd=liquidity_24h_usd
    )

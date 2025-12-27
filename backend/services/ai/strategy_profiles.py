"""
Strategy Profiles for Meta-Strategy Selector

Defines all available trading strategies with their TP/SL parameters,
risk profiles, and applicability criteria.

Each strategy is a complete trading profile that can be dynamically
selected based on market regime and symbol characteristics.

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyID(str, Enum):
    """Available trading strategies."""
    DEFENSIVE = "defensive"           # Conservative: tight stops, quick profits
    MODERATE = "moderate"             # Balanced: medium risk/reward
    MODERATE_AGGRESSIVE = "moderate_aggressive"  # Above moderate
    BALANCED_AGGRESSIVE = "balanced_aggressive"  # High risk/reward
    ULTRA_AGGRESSIVE = "ultra_aggressive"        # Maximum profit targets
    SCALP = "scalp"                   # Very tight TP/SL for range markets
    TREND_RIDER = "trend_rider"       # Wide stops, big targets for trends
    NONE = "none"                     # No trading (risk-off)


class AggressivenessLevel(str, Enum):
    """Strategy aggressiveness classification."""
    VERY_LOW = "very_low"    # SCALP, DEFENSIVE
    LOW = "low"              # MODERATE
    MEDIUM = "medium"        # MODERATE_AGGRESSIVE
    HIGH = "high"            # BALANCED_AGGRESSIVE
    VERY_HIGH = "very_high"  # ULTRA_AGGRESSIVE, TREND_RIDER


@dataclass
class StrategyProfile:
    """
    Complete strategy definition with TP/SL parameters and metadata.
    
    Attributes:
        strategy_id: Unique strategy identifier
        name: Human-readable name
        description: Strategy description
        
        # TP/SL Parameters (R-based)
        r_sl: Stop loss in R (ATR multiples)
        r_tp1: First take profit in R
        r_tp2: Second take profit in R
        r_tp3: Third take profit in R (optional)
        r_be_trigger: Move to breakeven at this R
        
        # Partial close fractions
        partial_close_tp1: Fraction to close at TP1 (0.0-1.0)
        partial_close_tp2: Fraction to close at TP2 (0.0-1.0)
        partial_close_tp3: Fraction to close at TP3 (0.0-1.0)
        
        # Trailing parameters
        trail_activation_r: Activate trailing at this R
        trail_distance_r: Trailing distance in R
        
        # Risk parameters
        aggressiveness: Strategy aggressiveness level
        risk_multiplier: Base risk adjustment (1.0 = normal)
        min_confidence_threshold: Minimum AI confidence required
        
        # Applicability
        suitable_for_trending: Works well in trending markets
        suitable_for_ranging: Works well in ranging markets
        suitable_for_high_vol: Works well in high volatility
        suitable_for_low_vol: Works well in low volatility
        suitable_symbol_tiers: Which symbol tiers this works for
        
        # Performance expectations
        expected_win_rate: Expected win rate (0.0-1.0)
        expected_avg_r: Expected average R per trade
        expected_risk_reward: Expected risk:reward ratio
    """
    
    strategy_id: StrategyID
    name: str
    description: str
    
    # TP/SL Parameters
    r_sl: float
    r_tp1: float
    r_tp2: float
    r_tp3: float
    r_be_trigger: float
    
    # Partial closes
    partial_close_tp1: float = 0.5
    partial_close_tp2: float = 0.3
    partial_close_tp3: float = 0.2
    
    # Trailing
    trail_activation_r: float = 2.5
    trail_distance_r: float = 0.8
    
    # Risk parameters
    aggressiveness: AggressivenessLevel = AggressivenessLevel.MEDIUM
    risk_multiplier: float = 1.0
    min_confidence_threshold: float = 0.50
    
    # Applicability
    suitable_for_trending: bool = True
    suitable_for_ranging: bool = True
    suitable_for_high_vol: bool = True
    suitable_for_low_vol: bool = True
    suitable_symbol_tiers: List[str] = field(default_factory=lambda: ["main", "l1", "l2"])
    
    # Performance expectations (for documentation/monitoring)
    expected_win_rate: float = 0.50
    expected_avg_r: float = 1.5
    expected_risk_reward: float = 1.5
    
    def to_tpsl_config(self) -> Dict[str, float]:
        """
        Convert to TpslConfig-compatible dictionary.
        
        Returns:
            Dictionary with TP/SL parameters for Trading Profile
        """
        return {
            "atr_mult_sl": self.r_sl,
            "atr_mult_tp1": self.r_tp1,
            "atr_mult_tp2": self.r_tp2,
            "atr_mult_tp3": self.r_tp3,
            "atr_mult_be": self.r_be_trigger,
            "partial_close_tp1": self.partial_close_tp1,
            "partial_close_tp2": self.partial_close_tp2,
            "trail_activation_mult": self.trail_activation_r,
            "trail_dist_mult": self.trail_distance_r,
        }
    
    def is_suitable_for_regime(self, regime: str) -> bool:
        """
        Check if strategy is suitable for given regime.
        
        Args:
            regime: Market regime string
            
        Returns:
            True if strategy is suitable
        """
        regime = regime.lower()
        
        if "trend" in regime:
            return self.suitable_for_trending
        elif "range" in regime:
            return self.suitable_for_ranging
        elif "high_vol" in regime or "volatile" in regime:
            return self.suitable_for_high_vol
        elif "low_vol" in regime:
            return self.suitable_for_low_vol
        
        return True  # Default: suitable
    
    def is_suitable_for_symbol_tier(self, tier: str) -> bool:
        """
        Check if strategy is suitable for symbol tier.
        
        Args:
            tier: Symbol tier (main, l1, l2, etc.)
            
        Returns:
            True if strategy is suitable
        """
        return tier.lower() in [t.lower() for t in self.suitable_symbol_tiers]


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

STRATEGY_PROFILES: Dict[StrategyID, StrategyProfile] = {
    
    # DEFENSIVE: Quick profits, tight stops, conservative
    StrategyID.DEFENSIVE: StrategyProfile(
        strategy_id=StrategyID.DEFENSIVE,
        name="Defensive",
        description="Conservative strategy with tight stops and quick profit taking",
        r_sl=1.0,
        r_tp1=1.5,
        r_tp2=2.5,
        r_tp3=4.0,
        r_be_trigger=1.0,
        partial_close_tp1=0.5,
        partial_close_tp2=0.3,
        partial_close_tp3=0.2,
        trail_activation_r=2.5,
        trail_distance_r=0.8,
        aggressiveness=AggressivenessLevel.VERY_LOW,
        risk_multiplier=0.8,
        min_confidence_threshold=0.45,
        suitable_for_trending=True,
        suitable_for_ranging=True,
        suitable_for_high_vol=False,
        suitable_for_low_vol=True,
        suitable_symbol_tiers=["main", "l1", "l2"],
        expected_win_rate=0.55,
        expected_avg_r=1.8,
        expected_risk_reward=1.8,
    ),
    
    # MODERATE: Balanced approach
    StrategyID.MODERATE: StrategyProfile(
        strategy_id=StrategyID.MODERATE,
        name="Moderate",
        description="Balanced strategy with moderate risk/reward",
        r_sl=1.0,
        r_tp1=2.0,
        r_tp2=3.5,
        r_tp3=5.5,
        r_be_trigger=1.2,
        partial_close_tp1=0.5,
        partial_close_tp2=0.3,
        partial_close_tp3=0.2,
        trail_activation_r=3.0,
        trail_distance_r=1.0,
        aggressiveness=AggressivenessLevel.LOW,
        risk_multiplier=1.0,
        min_confidence_threshold=0.48,
        suitable_for_trending=True,
        suitable_for_ranging=True,
        suitable_for_high_vol=True,
        suitable_for_low_vol=True,
        suitable_symbol_tiers=["main", "l1", "l2"],
        expected_win_rate=0.52,
        expected_avg_r=2.3,
        expected_risk_reward=2.3,
    ),
    
    # MODERATE AGGRESSIVE: Above moderate
    StrategyID.MODERATE_AGGRESSIVE: StrategyProfile(
        strategy_id=StrategyID.MODERATE_AGGRESSIVE,
        name="Moderate Aggressive",
        description="Above moderate strategy with good risk/reward",
        r_sl=1.0,
        r_tp1=2.5,
        r_tp2=4.0,
        r_tp3=6.0,
        r_be_trigger=1.5,
        partial_close_tp1=0.5,
        partial_close_tp2=0.3,
        partial_close_tp3=0.2,
        trail_activation_r=3.5,
        trail_distance_r=1.0,
        aggressiveness=AggressivenessLevel.MEDIUM,
        risk_multiplier=1.1,
        min_confidence_threshold=0.50,
        suitable_for_trending=True,
        suitable_for_ranging=False,
        suitable_for_high_vol=True,
        suitable_for_low_vol=False,
        suitable_symbol_tiers=["main", "l1"],
        expected_win_rate=0.50,
        expected_avg_r=3.0,
        expected_risk_reward=3.0,
    ),
    
    # BALANCED AGGRESSIVE: High risk/reward
    StrategyID.BALANCED_AGGRESSIVE: StrategyProfile(
        strategy_id=StrategyID.BALANCED_AGGRESSIVE,
        name="Balanced Aggressive",
        description="Aggressive strategy with wider stops and bigger targets",
        r_sl=1.5,
        r_tp1=4.0,
        r_tp2=6.0,
        r_tp3=10.0,
        r_be_trigger=2.0,
        partial_close_tp1=0.5,
        partial_close_tp2=0.3,
        partial_close_tp3=0.2,
        trail_activation_r=5.0,
        trail_distance_r=1.2,
        aggressiveness=AggressivenessLevel.HIGH,
        risk_multiplier=1.2,
        min_confidence_threshold=0.52,
        suitable_for_trending=True,
        suitable_for_ranging=False,
        suitable_for_high_vol=True,
        suitable_for_low_vol=False,
        suitable_symbol_tiers=["main", "l1"],
        expected_win_rate=0.48,
        expected_avg_r=4.5,
        expected_risk_reward=4.5,
    ),
    
    # ULTRA AGGRESSIVE: Maximum profit targets (CURRENT DEFAULT)
    StrategyID.ULTRA_AGGRESSIVE: StrategyProfile(
        strategy_id=StrategyID.ULTRA_AGGRESSIVE,
        name="Ultra Aggressive",
        description="Maximum profit targeting with tight stops",
        r_sl=1.0,
        r_tp1=3.0,
        r_tp2=5.0,
        r_tp3=8.0,
        r_be_trigger=1.5,
        partial_close_tp1=0.5,
        partial_close_tp2=0.3,
        partial_close_tp3=0.2,
        trail_activation_r=4.0,
        trail_distance_r=1.0,
        aggressiveness=AggressivenessLevel.VERY_HIGH,
        risk_multiplier=1.0,
        min_confidence_threshold=0.53,
        suitable_for_trending=True,
        suitable_for_ranging=False,
        suitable_for_high_vol=True,
        suitable_for_low_vol=False,
        suitable_symbol_tiers=["main", "l1"],
        expected_win_rate=0.45,
        expected_avg_r=5.0,
        expected_risk_reward=5.0,
    ),
    
    # SCALP: Very tight TP/SL for range-bound markets
    StrategyID.SCALP: StrategyProfile(
        strategy_id=StrategyID.SCALP,
        name="Scalp",
        description="Quick scalping strategy for range-bound markets",
        r_sl=0.8,
        r_tp1=1.2,
        r_tp2=1.8,
        r_tp3=2.5,
        r_be_trigger=0.6,
        partial_close_tp1=0.6,
        partial_close_tp2=0.3,
        partial_close_tp3=0.1,
        trail_activation_r=1.5,
        trail_distance_r=0.6,
        aggressiveness=AggressivenessLevel.VERY_LOW,
        risk_multiplier=0.7,
        min_confidence_threshold=0.55,
        suitable_for_trending=False,
        suitable_for_ranging=True,
        suitable_for_high_vol=False,
        suitable_for_low_vol=True,
        suitable_symbol_tiers=["main", "l1"],
        expected_win_rate=0.60,
        expected_avg_r=1.3,
        expected_risk_reward=1.5,
    ),
    
    # TREND RIDER: Wide stops, huge targets for strong trends
    StrategyID.TREND_RIDER: StrategyProfile(
        strategy_id=StrategyID.TREND_RIDER,
        name="Trend Rider",
        description="Ride strong trends with wide stops and huge profit targets",
        r_sl=2.0,
        r_tp1=5.0,
        r_tp2=8.0,
        r_tp3=12.0,
        r_be_trigger=2.5,
        partial_close_tp1=0.3,
        partial_close_tp2=0.3,
        partial_close_tp3=0.4,
        trail_activation_r=6.0,
        trail_distance_r=1.5,
        aggressiveness=AggressivenessLevel.VERY_HIGH,
        risk_multiplier=1.3,
        min_confidence_threshold=0.60,
        suitable_for_trending=True,
        suitable_for_ranging=False,
        suitable_for_high_vol=True,
        suitable_for_low_vol=False,
        suitable_symbol_tiers=["main", "l1"],
        expected_win_rate=0.40,
        expected_avg_r=6.5,
        expected_risk_reward=6.5,
    ),
    
    # NONE: Risk-off, no trading
    StrategyID.NONE: StrategyProfile(
        strategy_id=StrategyID.NONE,
        name="None (Risk Off)",
        description="No trading allowed - risk-off mode",
        r_sl=1.0,
        r_tp1=1.0,
        r_tp2=1.0,
        r_tp3=1.0,
        r_be_trigger=1.0,
        aggressiveness=AggressivenessLevel.VERY_LOW,
        risk_multiplier=0.0,
        min_confidence_threshold=1.0,  # Impossible threshold
        suitable_for_trending=False,
        suitable_for_ranging=False,
        suitable_for_high_vol=False,
        suitable_for_low_vol=False,
        suitable_symbol_tiers=[],
        expected_win_rate=0.0,
        expected_avg_r=0.0,
        expected_risk_reward=0.0,
    ),
}


# ============================================================================
# PUBLIC API
# ============================================================================

def get_strategy_profile(strategy_id: StrategyID | str) -> StrategyProfile:
    """
    Get strategy profile by ID.
    
    Args:
        strategy_id: Strategy identifier
        
    Returns:
        StrategyProfile object
        
    Raises:
        ValueError: If strategy not found
        
    Example:
        >>> profile = get_strategy_profile(StrategyID.ULTRA_AGGRESSIVE)
        >>> print(f"TP1: {profile.r_tp1}R, TP2: {profile.r_tp2}R")
        TP1: 3.0R, TP2: 5.0R
    """
    if isinstance(strategy_id, str):
        try:
            strategy_id = StrategyID(strategy_id.lower())
        except ValueError:
            raise ValueError(f"Unknown strategy: {strategy_id}")
    
    if strategy_id not in STRATEGY_PROFILES:
        raise ValueError(f"Strategy profile not found: {strategy_id}")
    
    return STRATEGY_PROFILES[strategy_id]


def get_all_strategy_ids() -> List[StrategyID]:
    """Get list of all available strategy IDs."""
    return list(STRATEGY_PROFILES.keys())


def get_strategies_for_regime(regime: str) -> List[StrategyProfile]:
    """
    Get all strategies suitable for given regime.
    
    Args:
        regime: Market regime string
        
    Returns:
        List of suitable StrategyProfile objects
        
    Example:
        >>> strategies = get_strategies_for_regime("TREND_UP")
        >>> for s in strategies:
        ...     print(s.name, s.expected_risk_reward)
    """
    return [
        profile for profile in STRATEGY_PROFILES.values()
        if profile.is_suitable_for_regime(regime) and profile.strategy_id != StrategyID.NONE
    ]


def get_strategies_for_symbol_tier(tier: str) -> List[StrategyProfile]:
    """
    Get all strategies suitable for given symbol tier.
    
    Args:
        tier: Symbol tier (main, l1, l2, etc.)
        
    Returns:
        List of suitable StrategyProfile objects
    """
    return [
        profile for profile in STRATEGY_PROFILES.values()
        if profile.is_suitable_for_symbol_tier(tier) and profile.strategy_id != StrategyID.NONE
    ]


def get_default_strategy() -> StrategyProfile:
    """Get default strategy (ULTRA_AGGRESSIVE as currently configured)."""
    return STRATEGY_PROFILES[StrategyID.ULTRA_AGGRESSIVE]


def list_all_strategies() -> Dict[str, Dict]:
    """
    Get overview of all strategies.
    
    Returns:
        Dictionary with strategy summaries
        
    Example:
        >>> strategies = list_all_strategies()
        >>> for sid, info in strategies.items():
        ...     print(f"{info['name']}: {info['risk_reward']}")
    """
    result = {}
    for sid, profile in STRATEGY_PROFILES.items():
        result[sid.value] = {
            "name": profile.name,
            "description": profile.description,
            "sl": f"{profile.r_sl}R",
            "tp1": f"{profile.r_tp1}R",
            "tp2": f"{profile.r_tp2}R",
            "tp3": f"{profile.r_tp3}R",
            "aggressiveness": profile.aggressiveness.value,
            "risk_reward": profile.expected_risk_reward,
            "win_rate": profile.expected_win_rate,
            "suitable_trending": profile.suitable_for_trending,
            "suitable_ranging": profile.suitable_for_ranging,
        }
    return result


if __name__ == "__main__":
    # Demo
    print("🎯 AVAILABLE TRADING STRATEGIES\n")
    print(f"{'Strategy':<25} {'SL':<6} {'TP1':<6} {'TP2':<6} {'TP3':<6} {'R:R':<6} {'WR%':<6} {'Trending':<10} {'Ranging'}")
    print("=" * 100)
    
    for sid, profile in STRATEGY_PROFILES.items():
        if sid == StrategyID.NONE:
            continue
        
        print(
            f"{profile.name:<25} "
            f"{profile.r_sl:<6.1f} "
            f"{profile.r_tp1:<6.1f} "
            f"{profile.r_tp2:<6.1f} "
            f"{profile.r_tp3:<6.1f} "
            f"{profile.expected_risk_reward:<6.1f} "
            f"{profile.expected_win_rate*100:<6.0f} "
            f"{'✅' if profile.suitable_for_trending else '❌':<10} "
            f"{'✅' if profile.suitable_for_ranging else '❌'}"
        )

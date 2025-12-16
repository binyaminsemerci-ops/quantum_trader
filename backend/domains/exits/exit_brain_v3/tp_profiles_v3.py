"""
Exit Brain v3 - TP Profile System

Profile-driven multi-leg take-profit and trailing configurations.
Supports regime-specific, symbol-specific, and strategy-specific profiles
with fallback hierarchy.

Profiles define:
- Multi-leg TP ladders (TP1/TP2/TP3) with hard vs soft exits
- Trailing configurations with tightening curves
- Per-regime optimizations (TREND/RANGE/VOLATILE/CHOP)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class TPKind(str, Enum):
    """TP leg execution type"""
    HARD = "HARD"  # Market order at exact trigger
    SOFT = "SOFT"  # Limit order, allows price improvement


class MarketRegime(str, Enum):
    """Market regime classification"""
    TREND = "TREND"          # Trending market (let profits run)
    RANGE = "RANGE"          # Range-bound / scalp mode
    VOLATILE = "VOLATILE"    # High volatility (tighter exits)
    CHOP = "CHOP"            # Choppy / indecisive (quick exits)
    NORMAL = "NORMAL"        # Default regime


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TPProfileLeg:
    """
    Single leg in TP ladder.
    
    r_multiple: Risk multiple (e.g., 1.0 = 1R, where R = SL distance)
                If you risk 2%, 1R = 2% profit, 2R = 4% profit
    size_fraction: Portion of position to exit (0.0-1.0)
    kind: HARD (market order) or SOFT (limit order)
    """
    r_multiple: float
    size_fraction: float
    kind: TPKind = TPKind.HARD
    
    def __post_init__(self):
        """Validate leg configuration"""
        if not (0.0 < self.size_fraction <= 1.0):
            raise ValueError(f"size_fraction must be 0.0-1.0, got {self.size_fraction}")
        if self.r_multiple <= 0:
            raise ValueError(f"r_multiple must be positive, got {self.r_multiple}")


@dataclass
class TrailingProfile:
    """
    Trailing stop configuration.
    
    callback_pct: Initial trailing callback % (e.g., 0.015 = 1.5%)
    activation_r: Start trailing when position reaches this R multiple
    tightening_curve: Optional steps to tighten callback as profit increases
                      Format: [(r_threshold, new_callback_pct), ...]
                      Example: [(2.0, 0.010), (4.0, 0.005)]
                      = at 2R tighten to 1%, at 4R tighten to 0.5%
    """
    callback_pct: float
    activation_r: float = 0.5  # Start trailing at 0.5R profit
    tightening_curve: List[tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and sort tightening curve"""
        if self.callback_pct <= 0:
            raise ValueError(f"callback_pct must be positive, got {self.callback_pct}")
        if self.activation_r < 0:
            raise ValueError(f"activation_r must be >= 0, got {self.activation_r}")
        
        # Sort tightening curve by r_threshold
        self.tightening_curve.sort(key=lambda x: x[0])


@dataclass
class TPProfile:
    """
    Complete TP strategy profile.
    
    name: Profile identifier
    tp_legs: Ordered list of TP legs (TP1, TP2, TP3, ...)
    trailing: Optional trailing configuration (typically for final leg)
    description: Human-readable description of profile purpose
    """
    name: str
    tp_legs: List[TPProfileLeg]
    trailing: Optional[TrailingProfile] = None
    description: str = ""
    
    def __post_init__(self):
        """Validate profile"""
        if not self.tp_legs:
            raise ValueError("TPProfile must have at least one TP leg")
        
        # Check that size fractions sum to <= 1.0
        total_size = sum(leg.size_fraction for leg in self.tp_legs)
        if total_size > 1.0:
            raise ValueError(f"TP leg sizes sum to {total_size}, must be <= 1.0")
    
    @property
    def total_tp_size(self) -> float:
        """Total position fraction covered by TP legs"""
        return sum(leg.size_fraction for leg in self.tp_legs)


@dataclass
class TPProfileMapping:
    """
    Profile selection key.
    
    Supports multi-level specificity:
    - (symbol, strategy_id, regime): Most specific
    - (symbol, strategy_id, ANY): Symbol + strategy default
    - (symbol, ANY, regime): Symbol + regime default
    - (ANY, strategy_id, regime): Strategy + regime default
    - (ANY, ANY, regime): Regime default
    - (ANY, ANY, ANY): Global default
    """
    symbol: str = "*"  # "*" = any symbol
    strategy_id: str = "*"  # "*" = any strategy
    regime: MarketRegime = MarketRegime.NORMAL
    
    def matches(self, symbol: str, strategy_id: str, regime: MarketRegime) -> bool:
        """Check if this mapping matches given parameters"""
        return (
            (self.symbol == "*" or self.symbol == symbol) and
            (self.strategy_id == "*" or self.strategy_id == strategy_id) and
            (self.regime == MarketRegime.NORMAL or self.regime == regime)
        )
    
    def specificity(self) -> int:
        """Calculate specificity score (higher = more specific)"""
        score = 0
        if self.symbol != "*":
            score += 100
        if self.strategy_id != "*":
            score += 10
        if self.regime != MarketRegime.NORMAL:
            score += 1
        return score


# ============================================================================
# Profile Definitions
# ============================================================================

# Default profiles for different market regimes

DEFAULT_TREND_PROFILE = TPProfile(
    name="TREND_DEFAULT",
    tp_legs=[
        TPProfileLeg(r_multiple=0.5, size_fraction=0.15, kind=TPKind.SOFT),   # TP1: 15% at 0.5R (lock early)
        TPProfileLeg(r_multiple=1.0, size_fraction=0.20, kind=TPKind.HARD),   # TP2: 20% at 1R (core target)
        TPProfileLeg(r_multiple=2.0, size_fraction=0.30, kind=TPKind.HARD),   # TP3: 30% at 2R (extended)
    ],
    trailing=TrailingProfile(
        callback_pct=0.020,  # 2% trailing for remaining 35%
        activation_r=1.5,    # Start trailing at 1.5R
        tightening_curve=[
            (3.0, 0.015),  # At 3R, tighten to 1.5%
            (5.0, 0.010),  # At 5R, tighten to 1.0%
        ]
    ),
    description="Trend-following: Let profits run with wide trailing"
)

DEFAULT_RANGE_PROFILE = TPProfile(
    name="RANGE_DEFAULT",
    tp_legs=[
        TPProfileLeg(r_multiple=0.2, size_fraction=0.35, kind=TPKind.SOFT),   # TP1: 35% at 0.2R (very quick profit)
        TPProfileLeg(r_multiple=0.4, size_fraction=0.35, kind=TPKind.HARD),   # TP2: 35% at 0.4R (core exit)
        TPProfileLeg(r_multiple=0.7, size_fraction=0.30, kind=TPKind.HARD),   # TP3: 30% at 0.7R (final exit)
    ],
    trailing=None,  # No trailing in range mode (quick exits)
    description="Range/Scalp: Very quick exits, lock profits early in consolidation"
)

DEFAULT_VOLATILE_PROFILE = TPProfile(
    name="VOLATILE_DEFAULT",
    tp_legs=[
        TPProfileLeg(r_multiple=0.4, size_fraction=0.25, kind=TPKind.SOFT),   # TP1: 25% at 0.4R
        TPProfileLeg(r_multiple=0.8, size_fraction=0.35, kind=TPKind.HARD),   # TP2: 35% at 0.8R
        TPProfileLeg(r_multiple=1.5, size_fraction=0.40, kind=TPKind.HARD),   # TP3: 40% at 1.5R
    ],
    trailing=TrailingProfile(
        callback_pct=0.025,  # Wider 2.5% trailing (avoid noise)
        activation_r=1.0,
        tightening_curve=[]  # No tightening in volatile mode
    ),
    description="Volatile: Wider stops, conservative targets"
)

DEFAULT_CHOP_PROFILE = TPProfile(
    name="CHOP_DEFAULT",
    tp_legs=[
        TPProfileLeg(r_multiple=0.15, size_fraction=0.40, kind=TPKind.SOFT),  # TP1: 40% at 0.15R (extremely quick)
        TPProfileLeg(r_multiple=0.35, size_fraction=0.40, kind=TPKind.HARD),  # TP2: 40% at 0.35R
        TPProfileLeg(r_multiple=0.6, size_fraction=0.20, kind=TPKind.HARD),   # TP3: 20% at 0.6R
    ],
    trailing=None,  # No trailing in choppy market
    description="Chop: Extremely quick exits, avoid prolonged exposure to whipsaws"
)

DEFAULT_NORMAL_PROFILE = TPProfile(
    name="NORMAL_DEFAULT",
    tp_legs=[
        TPProfileLeg(r_multiple=0.5, size_fraction=0.25, kind=TPKind.SOFT),   # TP1: 25% at 0.5R
        TPProfileLeg(r_multiple=1.0, size_fraction=0.25, kind=TPKind.HARD),   # TP2: 25% at 1R
        TPProfileLeg(r_multiple=2.0, size_fraction=0.50, kind=TPKind.HARD),   # TP3: 50% at 2R
    ],
    trailing=TrailingProfile(
        callback_pct=0.015,  # 1.5% trailing
        activation_r=1.0,
        tightening_curve=[(3.0, 0.010)]
    ),
    description="Normal: Balanced TP ladder with moderate trailing"
)

CHALLENGE_100_PROFILE = TPProfile(
    name="CHALLENGE_100",
    tp_legs=[
        TPProfileLeg(r_multiple=1.0, size_fraction=0.30, kind=TPKind.HARD),   # TP1: 30% at +1R
        # Remaining 70% is runner (managed by CHALLENGE_100 logic in executor)
    ],
    trailing=TrailingProfile(
        callback_pct=0.02,  # 2% trailing for runner (will be overridden by 2*ATR logic)
        activation_r=1.0,   # Start trailing immediately after TP1
        tightening_curve=[]
    ),
    description="$100 Challenge: Small losses, big winners - TP1 30% @ +1R, then runner with 2*ATR trailing"
)

# Global profile registry
PROFILE_REGISTRY: Dict[str, TPProfile] = {
    "TREND_DEFAULT": DEFAULT_TREND_PROFILE,
    "RANGE_DEFAULT": DEFAULT_RANGE_PROFILE,
    "VOLATILE_DEFAULT": DEFAULT_VOLATILE_PROFILE,
    "CHOP_DEFAULT": DEFAULT_CHOP_PROFILE,
    "NORMAL_DEFAULT": DEFAULT_NORMAL_PROFILE,
    "CHALLENGE_100": CHALLENGE_100_PROFILE,
}

# Profile mappings: (symbol, strategy_id, regime) -> profile_name
# NOTE: CHALLENGE_100 profile is selected dynamically via get_tp_and_trailing_profile()
# based on EXIT_BRAIN_PROFILE env flag, not via static mapping
PROFILE_MAPPINGS: List[tuple[TPProfileMapping, str]] = [
    # Regime-specific defaults (apply to all symbols/strategies)
    (TPProfileMapping(symbol="*", strategy_id="*", regime=MarketRegime.TREND), "TREND_DEFAULT"),
    (TPProfileMapping(symbol="*", strategy_id="*", regime=MarketRegime.RANGE), "RANGE_DEFAULT"),
    (TPProfileMapping(symbol="*", strategy_id="*", regime=MarketRegime.VOLATILE), "VOLATILE_DEFAULT"),
    (TPProfileMapping(symbol="*", strategy_id="*", regime=MarketRegime.CHOP), "CHOP_DEFAULT"),
    (TPProfileMapping(symbol="*", strategy_id="*", regime=MarketRegime.NORMAL), "NORMAL_DEFAULT"),
    
    # Symbol-specific overrides (example: BTC more conservative)
    # (TPProfileMapping(symbol="BTCUSDT", strategy_id="*", regime=MarketRegime.TREND), "BTC_TREND"),
    
    # Strategy-specific overrides (example: scalp strategy always uses RANGE profile)
    # (TPProfileMapping(symbol="*", strategy_id="SCALP_V2", regime=MarketRegime.TREND), "RANGE_DEFAULT"),
]


# ============================================================================
# Profile Lookup
# ============================================================================

def get_tp_and_trailing_profile(
    symbol: str,
    strategy_id: str,
    regime: MarketRegime
) -> tuple[TPProfile, Optional[TrailingProfile]]:
    """
    Get TP and trailing profiles for given context.
    
    **CHALLENGE_100 Profile Override:**
    If EXIT_BRAIN_PROFILE=CHALLENGE_100, returns CHALLENGE_100 profile
    regardless of regime. This takes precedence over all other mappings.
    
    Fallback hierarchy (most specific to least specific):
    1. (symbol, strategy_id, regime)
    2. (symbol, strategy_id, ANY)
    3. (symbol, ANY, regime)
    4. (ANY, strategy_id, regime)
    5. (ANY, ANY, regime)
    6. (ANY, ANY, NORMAL) - global default
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        strategy_id: Strategy identifier (e.g., "RL_V3", "TREND_FOLLOW")
        regime: Current market regime
        
    Returns:
        (TPProfile, TrailingProfile or None)
    """
    # PRIORITY 1: Check if CHALLENGE_100 profile is active (env flag)
    from backend.config.exit_mode import is_challenge_100_profile
    
    if is_challenge_100_profile():
        profile = CHALLENGE_100_PROFILE
        logger.info(
            f"[TP PROFILES] Using CHALLENGE_100 profile for {symbol} "
            f"(EXIT_BRAIN_PROFILE=CHALLENGE_100, ignoring regime={regime.value})"
        )
        return profile, profile.trailing
    
    # PRIORITY 2: Find all matching mappings
    candidates = []
    for mapping, profile_name in PROFILE_MAPPINGS:
        if mapping.matches(symbol, strategy_id, regime):
            candidates.append((mapping, profile_name))
    
    if not candidates:
        # Should never happen, but fallback to NORMAL_DEFAULT
        logger.warning(
            f"[TP PROFILES] No mapping found for {symbol}/{strategy_id}/{regime.value}, "
            f"using NORMAL_DEFAULT"
        )
        profile = DEFAULT_NORMAL_PROFILE
        return profile, profile.trailing
    
    # Sort by specificity (most specific first)
    candidates.sort(key=lambda x: x[0].specificity(), reverse=True)
    
    # Get most specific match
    mapping, profile_name = candidates[0]
    profile = PROFILE_REGISTRY.get(profile_name, DEFAULT_NORMAL_PROFILE)
    
    logger.info(
        f"[TP PROFILES] Selected '{profile_name}' for {symbol}/{strategy_id}/{regime.value} "
        f"(specificity={mapping.specificity()})"
    )
    
    return profile, profile.trailing


def register_custom_profile(
    profile: TPProfile,
    symbol: str = "*",
    strategy_id: str = "*",
    regime: MarketRegime = MarketRegime.NORMAL
):
    """
    Register a custom profile for specific context.
    
    Args:
        profile: TPProfile to register
        symbol: Symbol to apply to ("*" = all)
        strategy_id: Strategy to apply to ("*" = all)
        regime: Regime to apply to (NORMAL = default for all regimes)
    """
    # Add to registry
    PROFILE_REGISTRY[profile.name] = profile
    
    # Add mapping
    mapping = TPProfileMapping(symbol=symbol, strategy_id=strategy_id, regime=regime)
    PROFILE_MAPPINGS.insert(0, (mapping, profile.name))  # Insert at start for priority
    
    logger.info(
        f"[TP PROFILES] Registered custom profile '{profile.name}' for "
        f"{symbol}/{strategy_id}/{regime.value}"
    )


def get_profile_by_name(name: str) -> Optional[TPProfile]:
    """Get profile by name from registry"""
    return PROFILE_REGISTRY.get(name)


def list_available_profiles() -> List[str]:
    """List all registered profile names"""
    return list(PROFILE_REGISTRY.keys())


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_tp_price(
    entry_price: float,
    side: str,
    r_multiple: float,
    sl_distance_pct: float
) -> float:
    """
    Calculate TP price from R multiple.
    
    Args:
        entry_price: Entry price
        side: "LONG" or "SHORT"
        r_multiple: Risk multiple (1.0 = 1R)
        sl_distance_pct: SL distance as % (e.g., 0.025 = 2.5%)
        
    Returns:
        TP trigger price
    """
    tp_distance_pct = r_multiple * sl_distance_pct
    
    if side == "LONG":
        return entry_price * (1 + tp_distance_pct)
    else:  # SHORT
        return entry_price * (1 - tp_distance_pct)


def get_trailing_callback_for_profit(
    trailing_profile: TrailingProfile,
    current_r: float
) -> float:
    """
    Get appropriate trailing callback % based on current profit level.
    
    Applies tightening curve if position has reached higher R multiples.
    
    Args:
        trailing_profile: TrailingProfile with tightening curve
        current_r: Current profit in R multiples
        
    Returns:
        Appropriate callback % for current profit level
    """
    if not trailing_profile.tightening_curve:
        return trailing_profile.callback_pct
    
    # Find highest tightening threshold we've crossed
    active_callback = trailing_profile.callback_pct
    for r_threshold, new_callback in trailing_profile.tightening_curve:
        if current_r >= r_threshold:
            active_callback = new_callback
        else:
            break  # Curve is sorted, no need to check further
    
    return active_callback


def build_dynamic_tp_profile(ctx) -> Optional[TPProfile]:
    """
    Build a TP profile dynamically based on trade context.
    
    Uses leverage, position size, market regime, volatility and RL hints
    to construct an adaptive TPProfile compatible with the existing system.
    
    Args:
        ctx: ExitContext with position and market information
        
    Returns:
        TPProfile with dynamically calculated TP legs, or None if context insufficient
    """
    from backend.domains.exits.exit_brain_v3.dynamic_tp_calculator import calculate_dynamic_tp_levels
    
    try:
        # Calculate position size
        position_size_usd = ctx.size * ctx.entry_price
        
        # Get dynamic TP levels (returns list of (tp_pct, size_frac) tuples)
        dynamic_result = calculate_dynamic_tp_levels(
            symbol=ctx.symbol,
            position_size_usd=position_size_usd,
            leverage=ctx.leverage,
            volatility=ctx.volatility if ctx.volatility else 0.025,
            market_regime=ctx.market_regime,
            confidence=ctx.signal_confidence if ctx.signal_confidence else 0.75,
            unrealized_pnl_pct=ctx.unrealized_pnl_pct
        )
        
        # Convert dynamic TPs to TPProfile legs
        # Assume SL is 2.5% for R-multiple calculation
        sl_pct = ctx.max_loss_pct if ctx.max_loss_pct else 0.025
        
        tp_legs = []
        for tp_pct, size_frac in dynamic_result.tp_levels:
            r_multiple = tp_pct / sl_pct
            tp_legs.append(
                TPProfileLeg(
                    r_multiple=r_multiple,
                    size_fraction=size_frac,
                    kind=TPKind.HARD
                )
            )
        
        # Build profile
        profile = TPProfile(
            name=f"DYNAMIC_{ctx.symbol}_{ctx.leverage}x",
            tp_legs=tp_legs,
            trailing=None,  # Dynamic calculator doesn't include trailing yet
            description=f"Dynamic TP: {dynamic_result.reasoning}"
        )
        
        logger.info(
            f"[TP PROFILES] Built dynamic profile for {ctx.symbol}: "
            f"{len(tp_legs)} legs, confidence={dynamic_result.confidence:.1%}"
        )
        
        return profile
        
    except Exception as e:
        logger.error(
            f"[TP PROFILES] Failed to build dynamic profile for {ctx.symbol}: {e}",
            exc_info=True
        )
        return None

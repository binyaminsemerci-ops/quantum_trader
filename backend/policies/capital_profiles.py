"""
Capital Profile System for Quantum Trader v2.0

Defines risk profiles (Micro/Low/Normal/Aggressive) with explicit limits:
- Daily/weekly drawdown caps
- Single-trade risk limits
- Position limits
- Leverage caps

Part of EPIC-P10: Prompt 10 GO-LIVE Program
"""

from dataclasses import dataclass
from typing import Literal

# Type alias for profile names
CapitalProfileName = Literal["micro", "low", "normal", "aggressive"]


@dataclass
class CapitalProfile:
    """
    Capital profile defining risk limits for trading accounts.
    
    Attributes:
        name: Profile identifier
        max_daily_loss_pct: Maximum allowed daily loss (negative %)
        max_weekly_loss_pct: Maximum allowed weekly loss (negative %)
        max_single_trade_risk_pct: Maximum risk per single trade (% of capital)
        max_open_positions: Maximum concurrent open positions
        allowed_leverage: Maximum leverage multiplier (1 = spot only)
        description: Human-readable description
    """
    name: CapitalProfileName
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    max_single_trade_risk_pct: float
    max_open_positions: int
    allowed_leverage: int
    description: str = ""


# Conservative defaults for private multi-account trading
PROFILES: dict[CapitalProfileName, CapitalProfile] = {
    "micro": CapitalProfile(
        name="micro",
        max_daily_loss_pct=-0.5,      # -0.5% daily DD cap
        max_weekly_loss_pct=-2.0,     # -2.0% weekly DD cap
        max_single_trade_risk_pct=0.2, # 0.2% risk per trade
        max_open_positions=2,          # Max 2 concurrent positions
        allowed_leverage=1,            # Spot only (no leverage)
        description="Ultra-conservative profile for testing with real capital"
    ),
    
    "low": CapitalProfile(
        name="low",
        max_daily_loss_pct=-1.0,      # -1.0% daily DD cap
        max_weekly_loss_pct=-3.5,     # -3.5% weekly DD cap
        max_single_trade_risk_pct=0.5, # 0.5% risk per trade
        max_open_positions=3,          # Max 3 concurrent positions
        allowed_leverage=2,            # Up to 2x leverage
        description="Conservative profile for stable growth"
    ),
    
    "normal": CapitalProfile(
        name="normal",
        max_daily_loss_pct=-2.0,      # -2.0% daily DD cap
        max_weekly_loss_pct=-7.0,     # -7.0% weekly DD cap
        max_single_trade_risk_pct=1.0, # 1.0% risk per trade
        max_open_positions=5,          # Max 5 concurrent positions
        allowed_leverage=3,            # Up to 3x leverage
        description="Standard profile for normal trading operations"
    ),
    
    "aggressive": CapitalProfile(
        name="aggressive",
        max_daily_loss_pct=-3.5,      # -3.5% daily DD cap
        max_weekly_loss_pct=-12.0,    # -12.0% weekly DD cap
        max_single_trade_risk_pct=2.0, # 2.0% risk per trade
        max_open_positions=8,          # Max 8 concurrent positions
        allowed_leverage=5,            # Up to 5x leverage
        description="Aggressive profile for experienced accounts only"
    ),
}


# Progression ladder: testnet → micro → low → normal → aggressive
PROGRESSION_LADDER: list[CapitalProfileName] = ["micro", "low", "normal", "aggressive"]


def get_profile(name: CapitalProfileName) -> CapitalProfile:
    """
    Retrieve capital profile by name.
    
    Args:
        name: Profile name (micro/low/normal/aggressive)
        
    Returns:
        CapitalProfile configuration
        
    Raises:
        KeyError: If profile name not found
    """
    return PROFILES[name]


def list_profiles() -> list[CapitalProfile]:
    """Return all available capital profiles."""
    return [PROFILES[name] for name in PROGRESSION_LADDER]


def get_next_profile(current: CapitalProfileName) -> CapitalProfileName | None:
    """
    Get next profile in progression ladder.
    
    Args:
        current: Current profile name
        
    Returns:
        Next profile name, or None if already at max
        
    Example:
        >>> get_next_profile("micro")
        "low"
        >>> get_next_profile("aggressive")
        None
    """
    try:
        idx = PROGRESSION_LADDER.index(current)
    except ValueError:
        return None
    
    if idx + 1 < len(PROGRESSION_LADDER):
        return PROGRESSION_LADDER[idx + 1]
    return None


def get_previous_profile(current: CapitalProfileName) -> CapitalProfileName | None:
    """
    Get previous profile in progression ladder (for downgrades).
    
    Args:
        current: Current profile name
        
    Returns:
        Previous profile name, or None if already at minimum
    """
    try:
        idx = PROGRESSION_LADDER.index(current)
    except ValueError:
        return None
    
    if idx > 0:
        return PROGRESSION_LADDER[idx - 1]
    return None


# Promotion criteria (manual review required)
PROMOTION_CRITERIA = """
Manual promotion criteria for capital profiles:

MICRO → LOW:
  - No DD breach in 4 weeks
  - At least 20 trades executed
  - Win rate > 45%
  - Sharpe ratio > 0.5
  - Manual approval required

LOW → NORMAL:
  - No DD breach in 6 weeks
  - At least 50 trades executed
  - Win rate > 48%
  - Sharpe ratio > 0.8
  - Proven strategy performance
  - Manual approval required

NORMAL → AGGRESSIVE:
  - No DD breach in 8 weeks
  - At least 100 trades executed
  - Win rate > 50%
  - Sharpe ratio > 1.0
  - Extensive backtesting completed
  - Risk management expertise demonstrated
  - Manual approval required

DOWNGRADE TRIGGERS (automatic consideration):
  - DD breach of weekly limit
  - 2+ consecutive weeks negative PnL
  - Sharpe ratio drops below 0.3
  - Manual review → downgrade decision
"""


__all__ = [
    "CapitalProfileName",
    "CapitalProfile",
    "PROFILES",
    "PROGRESSION_LADDER",
    "PROMOTION_CRITERIA",
    "get_profile",
    "list_profiles",
    "get_next_profile",
    "get_previous_profile",
]

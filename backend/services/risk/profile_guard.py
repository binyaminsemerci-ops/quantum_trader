"""
Profile Guard - Capital Profile Risk Enforcement

Enforces capital profile limits before trade execution:
- Strategy whitelist/blacklist compliance
- Single-trade risk limits (TODO: wire to Global Risk v3)
- Daily/weekly drawdown limits (TODO: wire to Global Risk v3)
- Position count limits
- Leverage limits

Part of EPIC-P10: Prompt 10 GO-LIVE Program
"""

import logging
from typing import Optional
from backend.policies.capital_profiles import (
    CapitalProfileName,
    get_profile,
    CapitalProfile
)
from backend.policies.strategy_profile_policy import (
    is_strategy_allowed,
    StrategyNotAllowedError
)

logger = logging.getLogger(__name__)


class ProfileLimitViolationError(Exception):
    """Raised when trade would violate capital profile limits."""
    
    def __init__(self, profile: CapitalProfileName, limit_type: str, message: str):
        self.profile = profile
        self.limit_type = limit_type
        super().__init__(f"Profile '{profile}' {limit_type} limit: {message}")


def check_profile_allows_strategy(
    profile_name: CapitalProfileName,
    strategy_id: str
) -> None:
    """
    Check if strategy is allowed for capital profile.
    
    Args:
        profile_name: Capital profile name
        strategy_id: Strategy identifier
        
    Raises:
        StrategyNotAllowedError: If strategy not allowed for profile
        
    Example:
        >>> check_profile_allows_strategy("micro", "high_leverage_scalper")
        StrategyNotAllowedError: Strategy 'high_leverage_scalper' not allowed for profile 'micro'
    """
    if not is_strategy_allowed(profile_name, strategy_id):
        logger.warning(
            "Strategy blocked by capital profile",
            extra={
                "profile": profile_name,
                "strategy_id": strategy_id
            }
        )
        raise StrategyNotAllowedError(profile_name, strategy_id)
    
    logger.debug(
        "Strategy allowed for profile",
        extra={
            "profile": profile_name,
            "strategy_id": strategy_id
        }
    )


def check_leverage_limit(
    profile_name: CapitalProfileName,
    requested_leverage: int
) -> None:
    """
    Check if requested leverage is within profile limits.
    
    Args:
        profile_name: Capital profile name
        requested_leverage: Requested leverage multiplier
        
    Raises:
        ProfileLimitViolationError: If leverage exceeds limit
    """
    profile = get_profile(profile_name)
    
    if requested_leverage > profile.allowed_leverage:
        logger.warning(
            "Leverage limit exceeded",
            extra={
                "profile": profile_name,
                "requested_leverage": requested_leverage,
                "max_leverage": profile.allowed_leverage
            }
        )
        raise ProfileLimitViolationError(
            profile_name,
            "leverage",
            f"Requested {requested_leverage}x exceeds limit of {profile.allowed_leverage}x"
        )


def check_position_count_limit(
    profile_name: CapitalProfileName,
    current_positions: int,
    new_positions: int = 1
) -> None:
    """
    Check if opening new position(s) would exceed profile limit.
    
    Args:
        profile_name: Capital profile name
        current_positions: Current open position count
        new_positions: Number of new positions to open (default: 1)
        
    Raises:
        ProfileLimitViolationError: If would exceed max positions
    """
    profile = get_profile(profile_name)
    
    total_positions = current_positions + new_positions
    if total_positions > profile.max_open_positions:
        logger.warning(
            "Position count limit exceeded",
            extra={
                "profile": profile_name,
                "current_positions": current_positions,
                "new_positions": new_positions,
                "max_positions": profile.max_open_positions
            }
        )
        raise ProfileLimitViolationError(
            profile_name,
            "position_count",
            f"Total {total_positions} would exceed limit of {profile.max_open_positions}"
        )


def check_single_trade_risk(
    profile_name: CapitalProfileName,
    trade_risk_pct: float
) -> None:
    """
    Check if single trade risk is within profile limits.
    
    Args:
        profile_name: Capital profile name
        trade_risk_pct: Trade risk as percentage of capital (e.g., 0.5 = 0.5%)
        
    Raises:
        ProfileLimitViolationError: If risk exceeds limit
        
    Note:
        TODO: Wire to actual position sizing calculation
    """
    profile = get_profile(profile_name)
    
    if trade_risk_pct > profile.max_single_trade_risk_pct:
        logger.warning(
            "Single trade risk limit exceeded",
            extra={
                "profile": profile_name,
                "trade_risk_pct": trade_risk_pct,
                "max_risk_pct": profile.max_single_trade_risk_pct
            }
        )
        raise ProfileLimitViolationError(
            profile_name,
            "single_trade_risk",
            f"Risk {trade_risk_pct:.2f}% exceeds limit of {profile.max_single_trade_risk_pct:.2f}%"
        )


def check_daily_drawdown_limit(
    profile_name: CapitalProfileName,
    current_daily_pnl_pct: float
) -> None:
    """
    Check if current daily drawdown is within profile limits.
    
    Args:
        profile_name: Capital profile name
        current_daily_pnl_pct: Current daily PnL percentage (negative = loss)
        
    Raises:
        ProfileLimitViolationError: If drawdown exceeds limit
        
    Note:
        TODO: Wire to Global Risk v3 daily PnL metrics
    """
    profile = get_profile(profile_name)
    
    if current_daily_pnl_pct < profile.max_daily_loss_pct:
        logger.error(
            "Daily drawdown limit breached",
            extra={
                "profile": profile_name,
                "current_daily_pnl_pct": current_daily_pnl_pct,
                "max_daily_loss_pct": profile.max_daily_loss_pct
            }
        )
        raise ProfileLimitViolationError(
            profile_name,
            "daily_drawdown",
            f"Daily loss {current_daily_pnl_pct:.2f}% exceeds limit of {profile.max_daily_loss_pct:.2f}%"
        )


def check_weekly_drawdown_limit(
    profile_name: CapitalProfileName,
    current_weekly_pnl_pct: float
) -> None:
    """
    Check if current weekly drawdown is within profile limits.
    
    Args:
        profile_name: Capital profile name
        current_weekly_pnl_pct: Current weekly PnL percentage (negative = loss)
        
    Raises:
        ProfileLimitViolationError: If drawdown exceeds limit
        
    Note:
        TODO: Wire to Global Risk v3 weekly PnL metrics
    """
    profile = get_profile(profile_name)
    
    if current_weekly_pnl_pct < profile.max_weekly_loss_pct:
        logger.error(
            "Weekly drawdown limit breached",
            extra={
                "profile": profile_name,
                "current_weekly_pnl_pct": current_weekly_pnl_pct,
                "max_weekly_loss_pct": profile.max_weekly_loss_pct
            }
        )
        raise ProfileLimitViolationError(
            profile_name,
            "weekly_drawdown",
            f"Weekly loss {current_weekly_pnl_pct:.2f}% exceeds limit of {profile.max_weekly_loss_pct:.2f}%"
        )


def check_all_profile_limits(
    profile_name: CapitalProfileName,
    strategy_id: str,
    requested_leverage: int = 1,
    current_positions: int = 0,
    trade_risk_pct: Optional[float] = None,
    current_daily_pnl_pct: Optional[float] = None,
    current_weekly_pnl_pct: Optional[float] = None
) -> None:
    """
    Comprehensive profile limit check (all rules).
    
    Args:
        profile_name: Capital profile name
        strategy_id: Strategy identifier
        requested_leverage: Leverage multiplier (default: 1)
        current_positions: Current open position count (default: 0)
        trade_risk_pct: Single trade risk % (optional)
        current_daily_pnl_pct: Current daily PnL % (optional)
        current_weekly_pnl_pct: Current weekly PnL % (optional)
        
    Raises:
        StrategyNotAllowedError: If strategy not allowed
        ProfileLimitViolationError: If any limit violated
        
    Example:
        >>> check_all_profile_limits(
        ...     "micro",
        ...     "trend_follow_btc",
        ...     requested_leverage=1,
        ...     current_positions=1,
        ...     trade_risk_pct=0.15
        ... )
    """
    # 1. Strategy whitelist/blacklist
    check_profile_allows_strategy(profile_name, strategy_id)
    
    # 2. Leverage limit
    check_leverage_limit(profile_name, requested_leverage)
    
    # 3. Position count limit
    check_position_count_limit(profile_name, current_positions, new_positions=1)
    
    # 4. Single trade risk (if provided)
    if trade_risk_pct is not None:
        check_single_trade_risk(profile_name, trade_risk_pct)
    
    # 5. Daily drawdown (if provided)
    if current_daily_pnl_pct is not None:
        check_daily_drawdown_limit(profile_name, current_daily_pnl_pct)
    
    # 6. Weekly drawdown (if provided)
    if current_weekly_pnl_pct is not None:
        check_weekly_drawdown_limit(profile_name, current_weekly_pnl_pct)
    
    logger.info(
        "All profile limits passed",
        extra={
            "profile": profile_name,
            "strategy_id": strategy_id
        }
    )


__all__ = [
    "ProfileLimitViolationError",
    "check_profile_allows_strategy",
    "check_leverage_limit",
    "check_position_count_limit",
    "check_single_trade_risk",
    "check_daily_drawdown_limit",
    "check_weekly_drawdown_limit",
    "check_all_profile_limits",
]

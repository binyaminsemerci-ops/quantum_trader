"""
Strategy Profile Policy for Quantum Trader v2.0

Controls which strategies are allowed per capital profile:
- Whitelist: Explicitly allowed strategies
- Blacklist: Explicitly forbidden strategies
- Default: Allow all if no whitelist defined

Part of EPIC-P10: Prompt 10 GO-LIVE Program
"""

from typing import Optional
from backend.policies.capital_profiles import CapitalProfileName


class StrategyNotAllowedError(Exception):
    """Raised when strategy is not allowed for given capital profile."""
    
    def __init__(self, profile: CapitalProfileName, strategy_id: str):
        self.profile = profile
        self.strategy_id = strategy_id
        super().__init__(
            f"Strategy '{strategy_id}' not allowed for profile '{profile}'"
        )


# Strategy whitelist per profile (empty = allow all not blacklisted)
STRATEGIES_WHITELIST: dict[CapitalProfileName, set[str]] = {
    "micro": {
        # Only safest, proven strategies for micro profile
        # Example: "trend_follow_btc", "mean_reversion_eth"
    },
    "low": {
        # Add more strategies as they prove themselves
    },
    "normal": {
        # Most strategies allowed at normal level
    },
    "aggressive": {
        # All strategies including experimental ones
    },
}


# Strategy blacklist per profile (always takes precedence)
STRATEGIES_BLACKLIST: dict[CapitalProfileName, set[str]] = {
    "micro": {
        # Block high-risk strategies for micro profile
        # Example: "high_leverage_scalper", "grid_bot_5x"
    },
    "low": {
        # Block very aggressive strategies
        # Example: "high_leverage_scalper"
    },
    "normal": {
        # Block only extremely risky strategies
    },
    "aggressive": {
        # Minimal blacklist, most things allowed
    },
}


# Global blacklist (applies to ALL profiles)
GLOBAL_BLACKLIST: set[str] = set()  # Initialize as empty set, not dict
# Add strategies that are never allowed in production
# Example: GLOBAL_BLACKLIST.add("experimental_untested")


def is_strategy_allowed(
    profile: CapitalProfileName,
    strategy_id: str
) -> bool:
    """
    Check if strategy is allowed for given capital profile.
    
    Decision logic:
    1. If in GLOBAL_BLACKLIST → False
    2. If in profile's STRATEGIES_BLACKLIST → False
    3. If profile has whitelist AND strategy not in it → False
    4. Otherwise → True
    
    Args:
        profile: Capital profile name
        strategy_id: Strategy identifier
        
    Returns:
        True if strategy allowed, False otherwise
    """
    # Check global blacklist first
    if strategy_id in GLOBAL_BLACKLIST:
        return False
    
    # Check profile-specific blacklist
    if strategy_id in STRATEGIES_BLACKLIST.get(profile, set()):
        return False
    
    # If whitelist exists for this profile, strategy must be in it
    whitelist = STRATEGIES_WHITELIST.get(profile)
    if whitelist:
        return strategy_id in whitelist
    
    # No whitelist = allow all not blacklisted
    return True


def check_strategy_allowed(
    profile: CapitalProfileName,
    strategy_id: str
) -> None:
    """
    Check if strategy is allowed, raise exception if not.
    
    Args:
        profile: Capital profile name
        strategy_id: Strategy identifier
        
    Raises:
        StrategyNotAllowedError: If strategy not allowed for profile
    """
    if not is_strategy_allowed(profile, strategy_id):
        raise StrategyNotAllowedError(profile, strategy_id)


def get_allowed_strategies(
    profile: CapitalProfileName,
    all_strategies: Optional[set[str]] = None
) -> set[str]:
    """
    Get set of allowed strategies for profile.
    
    Args:
        profile: Capital profile name
        all_strategies: Optional set of all available strategies
                       (if None, returns whitelist or empty set)
        
    Returns:
        Set of allowed strategy IDs
    """
    whitelist = STRATEGIES_WHITELIST.get(profile)
    blacklist = STRATEGIES_BLACKLIST.get(profile, set())
    
    # If whitelist exists, use it
    if whitelist:
        # Remove blacklisted items from whitelist
        return whitelist - blacklist - GLOBAL_BLACKLIST
    
    # If no whitelist and no all_strategies provided, return empty set
    if all_strategies is None:
        return set()
    
    # Return all_strategies minus blacklists
    return all_strategies - blacklist - GLOBAL_BLACKLIST


def add_strategy_to_whitelist(
    profile: CapitalProfileName,
    strategy_id: str
) -> None:
    """
    Add strategy to profile's whitelist (runtime modification).
    
    Args:
        profile: Capital profile name
        strategy_id: Strategy identifier
    """
    if profile not in STRATEGIES_WHITELIST:
        STRATEGIES_WHITELIST[profile] = set()
    STRATEGIES_WHITELIST[profile].add(strategy_id)


def add_strategy_to_blacklist(
    profile: CapitalProfileName,
    strategy_id: str
) -> None:
    """
    Add strategy to profile's blacklist (runtime modification).
    
    Args:
        profile: Capital profile name
        strategy_id: Strategy identifier
    """
    if profile not in STRATEGIES_BLACKLIST:
        STRATEGIES_BLACKLIST[profile] = set()
    STRATEGIES_BLACKLIST[profile].add(strategy_id)


def remove_strategy_from_whitelist(
    profile: CapitalProfileName,
    strategy_id: str
) -> None:
    """Remove strategy from profile's whitelist."""
    if profile in STRATEGIES_WHITELIST:
        STRATEGIES_WHITELIST[profile].discard(strategy_id)


def remove_strategy_from_blacklist(
    profile: CapitalProfileName,
    strategy_id: str
) -> None:
    """Remove strategy from profile's blacklist."""
    if profile in STRATEGIES_BLACKLIST:
        STRATEGIES_BLACKLIST[profile].discard(strategy_id)


__all__ = [
    "StrategyNotAllowedError",
    "STRATEGIES_WHITELIST",
    "STRATEGIES_BLACKLIST",
    "GLOBAL_BLACKLIST",
    "is_strategy_allowed",
    "check_strategy_allowed",
    "get_allowed_strategies",
    "add_strategy_to_whitelist",
    "add_strategy_to_blacklist",
    "remove_strategy_from_whitelist",
    "remove_strategy_from_blacklist",
]

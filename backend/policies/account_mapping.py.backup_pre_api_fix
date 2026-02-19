"""
Account Mapping Policy

EPIC-MT-ACCOUNTS-001: Maps strategies to trading accounts.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Strategy ID → Account Name mapping
STRATEGY_ACCOUNT_MAP: Dict[str, str] = {
    # Examples (configure as needed):
    # "scalper_btc": "main_binance",
    # "swing_eth": "main_binance",
    # "nordic_spot": "main_firi",
    # "friend_1_strategy": "friend_1_firi",
}


def get_account_for_strategy(
    strategy_id: Optional[str],
    exchange_name: str
) -> str:
    """
    Get account name for a strategy.
    
    EPIC-MT-ACCOUNTS-001: Determines which trading account to use
    based on strategy ID and exchange name.
    
    Resolution order:
    1. Check STRATEGY_ACCOUNT_MAP for explicit mapping
    2. Fallback to default account for exchange ("main_<exchange>")
    
    Args:
        strategy_id: Strategy identifier (e.g., "scalper_btc")
        exchange_name: Exchange name (e.g., "binance", "firi")
    
    Returns:
        Account name to use
    
    Example:
        account = get_account_for_strategy("scalper_btc", "binance")
        # Returns "main_binance" if no explicit mapping
        
        # With mapping:
        STRATEGY_ACCOUNT_MAP["friend_strategy"] = "friend_1_binance"
        account = get_account_for_strategy("friend_strategy", "binance")
        # Returns "friend_1_binance"
    """
    # Priority 1: Explicit strategy mapping
    if strategy_id and strategy_id in STRATEGY_ACCOUNT_MAP:
        account_name = STRATEGY_ACCOUNT_MAP[strategy_id]
        logger.debug(
            "Using explicit strategy → account mapping",
            extra={
                "strategy_id": strategy_id,
                "account_name": account_name,
                "source": "strategy_map"
            }
        )
        return account_name
    
    # Priority 2: Default account for exchange
    default_account = f"main_{exchange_name}"
    logger.debug(
        "Using default account for exchange",
        extra={
            "strategy_id": strategy_id,
            "exchange": exchange_name,
            "account_name": default_account,
            "source": "default"
        }
    )
    return default_account


def set_strategy_account_mapping(mapping: Dict[str, str]) -> None:
    """
    Update strategy → account mapping.
    
    Args:
        mapping: Dictionary of strategy_id → account_name
    
    Example:
        set_strategy_account_mapping({
            "scalper_btc": "main_binance",
            "friend_1_strategy": "friend_1_firi"
        })
    """
    STRATEGY_ACCOUNT_MAP.update(mapping)
    logger.info(
        "Updated strategy → account mapping",
        extra={"mapping": mapping}
    )

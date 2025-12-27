"""
Exchange Routing Policy

EPIC-EXCH-ROUTING-001: Strategy → Exchange mapping policy.
Provides centralized logic for routing strategies to specific exchanges.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Strategy ID → Exchange Name mapping
# Add strategy mappings here to route specific strategies to specific exchanges
STRATEGY_EXCHANGE_MAP: Dict[str, str] = {
    # Example mappings (uncomment and customize as needed):
    # "scalper_btc": "bybit",
    # "swing_eth": "okx",
    # "hedge_arb": "kraken",
    # "nordic_spot": "firi",
}

# Allowed exchanges (validated before routing)
ALLOWED_EXCHANGES = [
    "binance",
    "bybit",
    "okx",
    "kucoin",
    "kraken",
    "firi",
]

# Default exchange (fallback when strategy not mapped)
DEFAULT_EXCHANGE = "binance"


def get_exchange_for_strategy(strategy_id: Optional[str]) -> str:
    """
    Get exchange name for a strategy.
    
    Resolution order:
    1. Check STRATEGY_EXCHANGE_MAP for explicit mapping
    2. Fallback to DEFAULT_EXCHANGE
    
    Args:
        strategy_id: Strategy identifier (e.g., "scalper_btc", "ai_ensemble")
    
    Returns:
        Exchange name (e.g., "binance", "bybit", "okx")
    
    Example:
        exchange = get_exchange_for_strategy("swing_eth")
        # Returns "okx" if mapped, else "binance"
    """
    if not strategy_id:
        logger.debug(
            "No strategy_id provided, using default exchange",
            extra={"default_exchange": DEFAULT_EXCHANGE}
        )
        return DEFAULT_EXCHANGE
    
    exchange = STRATEGY_EXCHANGE_MAP.get(strategy_id)
    
    if exchange:
        logger.info(
            "Strategy mapped to exchange",
            extra={
                "strategy_id": strategy_id,
                "exchange": exchange,
                "source": "strategy_map"
            }
        )
        return exchange
    
    logger.debug(
        "Strategy not in map, using default exchange",
        extra={
            "strategy_id": strategy_id,
            "default_exchange": DEFAULT_EXCHANGE
        }
    )
    return DEFAULT_EXCHANGE


def validate_exchange_name(exchange_name: str) -> str:
    """
    Validate exchange name against allowed list.
    
    Args:
        exchange_name: Exchange identifier to validate
    
    Returns:
        Validated exchange name, or DEFAULT_EXCHANGE if invalid
    
    Raises:
        Warning logged if exchange not in allowed list
    
    Example:
        validated = validate_exchange_name("bybit")  # Returns "bybit"
        validated = validate_exchange_name("unknown")  # Returns "binance", logs warning
    """
    if exchange_name in ALLOWED_EXCHANGES:
        return exchange_name
    
    logger.warning(
        "Invalid exchange name, falling back to default",
        extra={
            "requested_exchange": exchange_name,
            "allowed_exchanges": ALLOWED_EXCHANGES,
            "fallback": DEFAULT_EXCHANGE
        }
    )
    return DEFAULT_EXCHANGE


def set_strategy_exchange_mapping(mapping: Dict[str, str]) -> None:
    """
    Update strategy → exchange mapping.
    
    Allows runtime configuration of strategy routing.
    
    Args:
        mapping: Dict of strategy_id -> exchange_name
    
    Example:
        set_strategy_exchange_mapping({
            "scalper_btc": "bybit",
            "swing_eth": "okx",
        })
    """
    global STRATEGY_EXCHANGE_MAP
    
    STRATEGY_EXCHANGE_MAP.update(mapping)
    
    logger.info(
        "Strategy→exchange mapping updated",
        extra={"mapping_count": len(STRATEGY_EXCHANGE_MAP)}
    )


def get_current_mapping() -> Dict[str, str]:
    """
    Get current strategy → exchange mapping.
    
    Returns:
        Copy of STRATEGY_EXCHANGE_MAP
    """
    return STRATEGY_EXCHANGE_MAP.copy()

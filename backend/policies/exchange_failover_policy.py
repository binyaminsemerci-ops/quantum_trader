"""
Exchange Failover Policy

EPIC-EXCH-FAIL-001: Defines failover chains for multi-exchange routing.
When primary exchange is down/degraded, orders route to healthy fallback exchange.

Instrumented with Prometheus metrics for dashboard visibility (EPIC-STRESS-DASH-001).
"""

import logging
from typing import Dict, List, Sequence

# EPIC-STRESS-DASH-001: Prometheus metrics for dashboard visibility
try:
    from infra.metrics.metrics import exchange_failover_events_total
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# FAILOVER CHAIN CONFIGURATION
# ============================================================================

# Per-exchange failover chains (ordered by priority)
# First exchange in chain is always attempted first, then fallbacks
DEFAULT_FAILOVER_CHAIN: Dict[str, Sequence[str]] = {
    # Binance: High liquidity exchanges first
    "binance": ("binance", "bybit", "okx", "kucoin", "kraken", "firi"),
    
    # Bybit: Low latency exchanges first
    "bybit": ("bybit", "okx", "binance", "kucoin", "kraken", "firi"),
    
    # OKX: Derivatives-focused exchanges first
    "okx": ("okx", "bybit", "binance", "kucoin", "kraken", "firi"),
    
    # KuCoin: Altcoin-heavy exchanges first
    "kucoin": ("kucoin", "okx", "bybit", "binance", "kraken", "firi"),
    
    # Kraken: Fiat-friendly exchanges first
    "kraken": ("kraken", "binance", "firi", "bybit", "okx", "kucoin"),
    
    # Firi: Nordic exchanges first, then major exchanges
    "firi": ("firi", "binance", "kraken", "bybit", "okx", "kucoin"),
}


def get_failover_chain(primary_exchange: str) -> List[str]:
    """
    Get failover chain for a given primary exchange.
    
    Returns ordered list of exchanges to try when primary is down.
    First exchange in chain is the primary (preferred) exchange.
    
    Args:
        primary_exchange: Primary exchange name (e.g., "binance", "bybit")
    
    Returns:
        List of exchange names to try in order
        If primary not configured, returns single-item list [primary_exchange]
    
    Example:
        chain = get_failover_chain("binance")
        # Returns ["binance", "bybit", "okx", "kucoin", "kraken", "firi"]
        
        chain = get_failover_chain("unknown_exchange")
        # Returns ["unknown_exchange"] (no failover)
    """
    chain = DEFAULT_FAILOVER_CHAIN.get(primary_exchange)
    
    if not chain:
        logger.warning(
            "No failover chain configured for exchange",
            extra={"exchange": primary_exchange}
        )
        return [primary_exchange]
    
    return list(chain)


def set_failover_chain(exchange: str, chain: Sequence[str]) -> None:
    """
    Update failover chain for an exchange (runtime configuration).
    
    Args:
        exchange: Exchange name
        chain: New failover chain (ordered sequence of exchange names)
    
    Example:
        # Prioritize low-latency exchanges for scalper strategy
        set_failover_chain("binance", ["binance", "bybit", "okx"])
    """
    DEFAULT_FAILOVER_CHAIN[exchange] = tuple(chain)
    logger.info(
        "Updated failover chain",
        extra={"exchange": exchange, "chain": chain}
    )


# ============================================================================
# HEALTH CHECK HELPERS
# ============================================================================

async def get_exchange_health(exchange: str) -> Dict[str, any]:
    """
    Check current health status of an exchange.
    
    Performs lightweight health check using exchange's health() method.
    Does NOT use caching - always performs live check.
    
    Args:
        exchange: Exchange name (e.g., "binance", "bybit")
    
    Returns:
        Health status dict with keys:
        - status: "ok" | "degraded" | "down"
        - latency_ms: Response time in milliseconds
        - last_error: Error message if any
    
    Raises:
        Exception: If exchange client creation or health check fails
    
    Example:
        health = await get_exchange_health("binance")
        # {"status": "ok", "latency_ms": 45, "last_error": None}
    """
    from backend.integrations.exchanges.factory import get_exchange_client
    
    try:
        client = get_exchange_client(exchange)
        health = await client.health()
        return health
    except Exception as e:
        logger.warning(
            "Health check failed for exchange",
            extra={"exchange": exchange, "error": str(e)}
        )
        # Return pessimistic health status
        return {
            "status": "down",
            "latency_ms": 0,
            "last_error": str(e)
        }


def is_healthy(health: Dict[str, any]) -> bool:
    """
    Determine if exchange health status is acceptable for trading.
    
    Args:
        health: Health status dict from get_exchange_health()
    
    Returns:
        True if exchange is healthy (status == "ok")
        False if degraded or down
    
    Example:
        health = {"status": "ok", "latency_ms": 45, "last_error": None}
        is_healthy(health)  # True
        
        health = {"status": "degraded", "latency_ms": 500, "last_error": "Timeout"}
        is_healthy(health)  # False
    """
    return health.get("status") == "ok"


# ============================================================================
# FAILOVER SELECTION LOGIC
# ============================================================================

async def choose_exchange_with_failover(
    primary_exchange: str,
    default_exchange: str = "binance"
) -> str:
    """
    Choose best available exchange based on health checks and failover chain.
    
    Core failover logic:
    1. Get failover chain for primary exchange
    2. Try each exchange in order until one is healthy
    3. If all fail, return default_exchange anyway (allow execution to attempt)
    
    Args:
        primary_exchange: Preferred exchange (e.g., "bybit")
        default_exchange: Fallback if all exchanges down (default: "binance")
    
    Returns:
        Exchange name to use for trading
        Always returns a valid exchange name (never None)
    
    Example:
        # Binance healthy
        exchange = await choose_exchange_with_failover("binance", "binance")
        # Returns "binance"
        
        # Binance down, Bybit healthy
        exchange = await choose_exchange_with_failover("binance", "binance")
        # Returns "bybit" (first healthy in chain)
        
        # All exchanges down
        exchange = await choose_exchange_with_failover("binance", "binance")
        # Returns "binance" (default_exchange - let execution handle error)
    """
    chain = get_failover_chain(primary_exchange)
    
    # Ensure default_exchange is in chain (safety net)
    if default_exchange not in chain:
        chain = list(chain) + [default_exchange]
    
    # Try each exchange in failover chain
    for exchange in chain:
        try:
            health = await get_exchange_health(exchange)
            
            if is_healthy(health):
                if exchange != primary_exchange:
                    logger.warning(
                        "Failover activated - primary exchange unhealthy",
                        extra={
                            "primary": primary_exchange,
                            "selected": exchange,
                            "latency_ms": health.get("latency_ms"),
                        }
                    )
                    # EPIC-STRESS-DASH-001: Record failover metric
                    if METRICS_AVAILABLE and exchange_failover_events_total:
                        try:
                            exchange_failover_events_total.labels(
                                primary=primary_exchange,
                                selected=exchange,
                            ).inc()
                        except Exception as e:
                            logger.warning(f"Failed to record failover metric: {e}")
                else:
                    logger.debug(
                        "Using primary exchange",
                        extra={
                            "exchange": exchange,
                            "latency_ms": health.get("latency_ms"),
                        }
                    )
                
                return exchange
            else:
                logger.debug(
                    "Exchange unhealthy, trying next in chain",
                    extra={
                        "exchange": exchange,
                        "status": health.get("status"),
                        "error": health.get("last_error"),
                    }
                )
        
        except Exception as e:
            logger.warning(
                "Health check failed, trying next exchange",
                extra={"exchange": exchange, "error": str(e)}
            )
            continue
    
    # All exchanges failed - return default_exchange anyway
    # Let execution layer handle the error
    logger.error(
        "All exchanges in failover chain are unhealthy",
        extra={
            "primary": primary_exchange,
            "chain": chain,
            "fallback": default_exchange,
        }
    )
    
    return default_exchange

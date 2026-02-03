"""
Exit Ownership Enforcement - Single Exit Controller

Binary invariant:
- ONLY exitbrain_v3_5 (or designated EXIT_OWNER) can emit reduceOnly=true orders
- All other services attempting to close positions will be DENIED

This prevents multiple systems from interfering with harvest control.

Usage in services:
    from lib.exit_ownership import validate_exit_ownership
    
    # Before emitting reduceOnly order
    if not validate_exit_ownership(source="my_service", symbol=symbol):
        logger.error("DENY_NOT_EXIT_OWNER")
        return  # SKIP - not authorized
"""

import os
import logging

logger = logging.getLogger(__name__)

# Designated exit owner (single source of truth)
EXIT_OWNER = os.getenv("QUANTUM_EXIT_OWNER", "exitbrain_v3_5")

# Services explicitly allowed to close positions (for emergency/manual interventions)
EXIT_ALLOWED_SERVICES = set([
    EXIT_OWNER,
    "manual_close",           # Manual emergency closes
    "emergency_stop",         # System-wide emergency stop
    "reconcile_engine"        # Position reconciliation (ledger sync)
])


def validate_exit_ownership(source: str, symbol: str = None) -> bool:
    """
    Validate if source is authorized to emit reduceOnly=true orders.
    
    Args:
        source: Service identifier (e.g., "harvest_publisher", "exitbrain_v3_5")
        symbol: Optional symbol for logging context
    
    Returns:
        True if authorized, False if DENY
    """
    if source in EXIT_ALLOWED_SERVICES:
        return True
    
    # DENY - log violation
    symbol_str = f" for {symbol}" if symbol else ""
    logger.error(
        f"DENY_NOT_EXIT_OWNER: {source} attempted to emit reduceOnly order{symbol_str} "
        f"(only {EXIT_OWNER} is authorized)"
    )
    
    return False


def get_exit_owner() -> str:
    """Get designated exit owner identifier"""
    return EXIT_OWNER


def is_exit_owner(source: str) -> bool:
    """Check if source is the designated exit owner (primary controller)"""
    return source == EXIT_OWNER

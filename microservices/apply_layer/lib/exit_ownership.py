"""
Exit Ownership Module
=====================
Enforces single-controller exit discipline:
  only EXIT_OWNER is permitted to send CLOSE signals through apply_layer.

Usage in apply_layer/main.py (lines ~1943-1964):
    if EXIT_OWNERSHIP_ENABLED:
        if plan.source != EXIT_OWNER:
            → DENY_NOT_EXIT_OWNER

To override the authorized service, set env var:
    EXIT_OWNER_SERVICE=exitbrain_v3_5  (default)
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---- Authorised exit controller ----
# Only one service may send CLOSE / FULL_CLOSE_PROPOSED plans.
# Set EXIT_OWNER_SERVICE=<name> to change from the default.
EXIT_OWNER: str = os.getenv("EXIT_OWNER_SERVICE", "exitbrain_v3_5")

logger.info(f"[exit_ownership] EXIT_OWNER={EXIT_OWNER!r} (override via EXIT_OWNER_SERVICE env)")


def validate_exit_ownership(
    redis_client,          # redis.Redis instance (passed in, no hard dependency)
    symbol: str,
    source: str,
    plan_id: Optional[str] = None,
) -> bool:
    """
    Validate that *source* is the authorised exit controller.

    Args:
        redis_client : active Redis connection (unused for basic check, available
                       for future per-symbol ownership locks)
        symbol       : trading symbol, e.g. "BTCUSDT"
        source       : value of plan.source field
        plan_id      : optional plan identifier for logging

    Returns:
        True  → source is EXIT_OWNER, close is authorised
        False → source is NOT EXIT_OWNER, close is denied
    """
    authorised = source == EXIT_OWNER
    if authorised:
        logger.debug(
            f"[exit_ownership] ALLOW symbol={symbol} source={source!r} plan_id={plan_id}"
        )
    else:
        logger.warning(
            f"[exit_ownership] DENY symbol={symbol} source={source!r} "
            f"expected={EXIT_OWNER!r} plan_id={plan_id}"
        )
    return authorised

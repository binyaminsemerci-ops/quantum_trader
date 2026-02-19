"""
Per-Account Risk Limits

EPIC-MT-ACCOUNTS-001: Placeholder for per-account risk controls.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def check_account_limits(account_name: str, order_request: Any) -> None:
    """
    Check per-account risk limits.
    
    EPIC-MT-ACCOUNTS-001: Placeholder for account-specific risk controls.
    
    Future implementation will check:
    - Per-account exposure caps
    - Per-account position limits
    - Per-account daily drawdown limits
    - Per-account max leverage
    
    Args:
        account_name: Trading account name
        order_request: Order request to validate
    
    Returns:
        None (raises exception if limits violated)
    
    TODO:
    - [ ] Implement per-account exposure tracking
    - [ ] Add account-specific limits to PolicyStore/AccountConfig
    - [ ] Integrate with Global Risk v3 (EPIC-RISK3-001)
    - [ ] Add per-account PnL/DD tracking
    
    Example:
        # Future usage
        check_account_limits("main_binance", order_request)
        # Raises RiskLimitViolation if account exposure exceeded
    """
    logger.debug(
        "Account limits check (NO-OP)",
        extra={
            "account_name": account_name,
            "order": getattr(order_request, "symbol", None)
        }
    )
    
    # NO-OP for now - real implementation in future EPIC
    # When implemented, raise exception if limits violated:
    # if account_exposure > account_limit:
    #     raise RiskLimitViolation(f"Account {account_name} exposure exceeded")
    
    return

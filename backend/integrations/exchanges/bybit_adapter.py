"""
Bybit Exchange Adapter (Skeleton)

EPIC-EXCH-001: Placeholder for future Bybit implementation.
All methods raise NotImplementedError.
"""

import logging
from typing import List, Optional

from backend.integrations.exchanges.base import IExchangeClient, ExchangeAPIError
from backend.integrations.exchanges.models import (
    OrderRequest,
    OrderResult,
    CancelResult,
    Position,
    Balance,
    Kline,
)

logger = logging.getLogger(__name__)


class BybitAdapter:
    """
    Bybit exchange adapter (skeleton implementation).
    
    Placeholder for future Bybit API integration.
    All methods raise NotImplementedError.
    
    Future implementation will integrate:
    - Bybit V5 API (unified account)
    - USDT perpetuals
    - Inverse perpetuals (optional)
    - Authentication & rate limiting
    
    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
        testnet: Whether to use testnet (default: False)
    
    Example:
        adapter = BybitAdapter(api_key, api_secret, testnet=True)
        # NotImplementedError - coming soon!
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange_name = "bybit"
        
        logger.info(
            "BybitAdapter initialized (skeleton - not functional)",
            extra={"exchange": "bybit", "testnet": testnet}
        )
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order on Bybit (not implemented)."""
        raise NotImplementedError(
            "Bybit adapter not yet implemented. "
            "Future implementation will support Bybit V5 API."
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """Cancel order on Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions from Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balances from Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """Get candlestick data from Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """Get order status from Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol on Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    async def close_position(self, symbol: str) -> OrderResult:
        """Close position on Bybit (not implemented)."""
        raise NotImplementedError("Bybit adapter not yet implemented.")
    
    def get_exchange_name(self) -> str:
        """Return exchange name."""
        return self.exchange_name

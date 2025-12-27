"""
OKX Exchange Adapter (Skeleton)

EPIC-EXCH-001: Placeholder for future OKX implementation.
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


class OKXAdapter:
    """
    OKX exchange adapter (skeleton implementation).
    
    Placeholder for future OKX API integration.
    All methods raise NotImplementedError.
    
    Future implementation will integrate:
    - OKX V5 API (unified account)
    - USDT-margined futures
    - Coin-margined futures (optional)
    - Authentication & rate limiting
    
    Args:
        api_key: OKX API key
        api_secret: OKX API secret
        passphrase: OKX API passphrase
        testnet: Whether to use testnet (default: False)
    
    Example:
        adapter = OKXAdapter(api_key, api_secret, passphrase, testnet=True)
        # NotImplementedError - coming soon!
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        testnet: bool = False
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self.exchange_name = "okx"
        
        logger.info(
            "OKXAdapter initialized (skeleton - not functional)",
            extra={"exchange": "okx", "testnet": testnet}
        )
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order on OKX (not implemented)."""
        raise NotImplementedError(
            "OKX adapter not yet implemented. "
            "Future implementation will support OKX V5 API."
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """Cancel order on OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions from OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balances from OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """Get candlestick data from OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """Get order status from OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol on OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    async def close_position(self, symbol: str) -> OrderResult:
        """Close position on OKX (not implemented)."""
        raise NotImplementedError("OKX adapter not yet implemented.")
    
    def get_exchange_name(self) -> str:
        """Return exchange name."""
        return self.exchange_name

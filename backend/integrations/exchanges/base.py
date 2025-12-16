"""
Exchange Client Interface

EPIC-EXCH-001: Protocol (abstract interface) that all exchange adapters must implement.
Defines the contract for trading operations across all exchanges.
"""

from typing import Protocol, List, Optional
from backend.integrations.exchanges.models import (
    OrderRequest,
    OrderResult,
    CancelResult,
    Position,
    Balance,
    Kline,
)


class IExchangeClient(Protocol):
    """
    Exchange client protocol.
    
    All exchange adapters (Binance, Bybit, OKX) must implement this interface.
    This ensures consistent API across all exchanges.
    
    Usage:
        client: IExchangeClient = get_exchange_client(ExchangeType.BINANCE, config)
        result = await client.place_order(order_request)
    """
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place a new order.
        
        Args:
            request: Order request with symbol, side, type, quantity, etc.
        
        Returns:
            OrderResult with order ID, status, filled quantity, etc.
        
        Raises:
            ExchangeAPIError: If order placement fails
            ValidationError: If request is invalid
        """
        ...
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """
        Cancel an existing order.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            order_id: Exchange order ID
        
        Returns:
            CancelResult with success status and final order state
        
        Raises:
            ExchangeAPIError: If cancellation fails
        """
        ...
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions (futures).
        
        Args:
            symbol: Optional symbol filter (e.g., BTCUSDT)
                    If None, returns all positions
        
        Returns:
            List of Position objects with current P&L, leverage, etc.
        
        Raises:
            ExchangeAPIError: If position fetch fails
        """
        ...
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """
        Get account balances.
        
        Args:
            asset: Optional asset filter (e.g., USDT)
                   If None, returns all balances
        
        Returns:
            List of Balance objects with free, locked, total amounts
        
        Raises:
            ExchangeAPIError: If balance fetch fails
        """
        ...
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """
        Get historical candlestick/kline data.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Maximum number of candles (default 500, max 1000)
            start_time: Optional start timestamp (ms)
            end_time: Optional end timestamp (ms)
        
        Returns:
            List of Kline objects with OHLCV data
        
        Raises:
            ExchangeAPIError: If kline fetch fails
        """
        ...
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """
        Get status of an existing order.
        
        Args:
            symbol: Trading pair
            order_id: Exchange order ID
        
        Returns:
            OrderResult with current order status
        
        Raises:
            ExchangeAPIError: If order query fails
        """
        ...
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol (futures).
        
        Args:
            symbol: Trading pair
            leverage: Leverage value (1-125 depending on exchange)
        
        Returns:
            True if successful
        
        Raises:
            ExchangeAPIError: If leverage change fails
        """
        ...
    
    async def close_position(self, symbol: str) -> OrderResult:
        """
        Close an open position (market order in opposite direction).
        
        Args:
            symbol: Trading pair
        
        Returns:
            OrderResult of closing order
        
        Raises:
            ExchangeAPIError: If position close fails
        """
        ...
    
    def get_exchange_name(self) -> str:
        """
        Get exchange name.
        
        Returns:
            Exchange name ("binance", "bybit", "okx", etc.)
        """
        ...


# ============================================================================
# HELPER TYPES
# ============================================================================

class ExchangeAPIError(Exception):
    """
    Base exception for exchange API errors.
    
    Wraps exchange-specific errors into a common format.
    """
    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        exchange: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.code = code
        self.exchange = exchange
        self.original_error = original_error
        super().__init__(message)
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.exchange:
            parts.append(f"exchange={self.exchange}")
        if self.code:
            parts.append(f"code={self.code}")
        return f"ExchangeAPIError({', '.join(parts)})"

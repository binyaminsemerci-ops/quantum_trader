"""
KuCoin Exchange Adapter

EPIC-EXCH-002: KuCoin Futures integration implementing IExchangeClient.
Supports USDT-margined perpetual futures trading.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import time
from decimal import Decimal
from typing import List, Optional, Dict, Any

import httpx

from backend.integrations.exchanges.base import IExchangeClient, ExchangeAPIError
from backend.integrations.exchanges.models import (
    OrderRequest,
    OrderResult,
    CancelResult,
    Position,
    Balance,
    Kline,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
)

logger = logging.getLogger(__name__)


class KuCoinAdapter:
    """
    KuCoin Futures adapter implementing IExchangeClient.
    
    Uses KuCoin Futures API v1 with HMAC-SHA256 authentication.
    Supports USDT-margined perpetual contracts.
    
    Args:
        api_key: KuCoin API key
        api_secret: KuCoin API secret
        passphrase: KuCoin API passphrase
        testnet: Use sandbox environment (default: False)
    
    Example:
        adapter = KuCoinAdapter(
            api_key=os.getenv("KUCOIN_API_KEY"),
            api_secret=os.getenv("KUCOIN_API_SECRET"),
            passphrase=os.getenv("KUCOIN_PASSPHRASE"),
            testnet=False
        )
        result = await adapter.place_order(order_request)
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
        self.exchange_name = "kucoin"
        
        # Base URLs
        if testnet:
            self.base_url = "https://api-sandbox-futures.kucoin.com"
        else:
            self.base_url = "https://api-futures.kucoin.com"
        
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        logger.info(
            f"KuCoinAdapter initialized (testnet={testnet})",
            extra={"exchange": "kucoin", "testnet": testnet}
        )
    
    # ========================================================================
    # Authentication & Signing
    # ========================================================================
    
    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        endpoint: str,
        body: str = ""
    ) -> str:
        """
        Generate HMAC-SHA256 signature for KuCoin API.
        
        Signature format: base64(hmac_sha256(timestamp + method + endpoint + body, api_secret))
        
        Args:
            timestamp: Unix timestamp in milliseconds (string)
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path (e.g., /api/v1/orders)
            body: Request body (JSON string, empty for GET)
        
        Returns:
            Base64-encoded signature
        """
        # KuCoin signature: timestamp + method + endpoint + body
        sign_str = timestamp + method + endpoint + body
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _generate_passphrase_signature(self) -> str:
        """
        Generate encrypted passphrase for KuCoin API.
        
        Passphrase must be encrypted with api_secret using HMAC-SHA256.
        
        Returns:
            Base64-encoded encrypted passphrase
        """
        passphrase_sig = hmac.new(
            self.api_secret.encode('utf-8'),
            self.passphrase.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(passphrase_sig).decode('utf-8')
    
    def _build_headers(
        self,
        method: str,
        endpoint: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Build authenticated request headers for KuCoin API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            body: Request body (empty for GET)
        
        Returns:
            Headers dict with signature and authentication
        """
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, endpoint, body)
        passphrase = self._generate_passphrase_signature()
        
        return {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",  # API v2 (encrypted passphrase)
            "Content-Type": "application/json"
        }
    
    # ========================================================================
    # IExchangeClient Implementation
    # ========================================================================
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place order on KuCoin Futures.
        
        Maps OrderRequest -> KuCoin API format -> OrderResult.
        
        API: POST /api/v1/orders
        """
        try:
            endpoint = "/api/v1/orders"
            
            # Map to KuCoin order format
            order_data = {
                "symbol": request.symbol,  # e.g., XBTUSDTM
                "side": "buy" if request.side == OrderSide.BUY else "sell",
                "lever": request.leverage or 1,
                "type": self._map_order_type(request.order_type),
                "size": int(request.quantity),  # KuCoin uses contracts (integer)
            }
            
            # Add price for limit orders
            if request.order_type == OrderType.LIMIT and request.price:
                order_data["price"] = str(request.price)
            
            # Add stop price for stop orders
            if request.stop_price:
                order_data["stopPrice"] = str(request.stop_price)
            
            # Reduce-only flag
            if request.reduce_only:
                order_data["reduceOnly"] = True
            
            # Client order ID
            if request.client_order_id:
                order_data["clientOid"] = request.client_order_id
            
            body = httpx._utils.json.dumps(order_data)
            headers = self._build_headers("POST", endpoint, body)
            
            # Execute request
            url = self.base_url + endpoint
            response = await self.http_client.post(url, headers=headers, content=body)
            
            if response.status_code != 200:
                raise ExchangeAPIError(
                    message=f"KuCoin order failed: {response.text}",
                    code=response.status_code,
                    exchange="kucoin"
                )
            
            data = response.json()
            
            if data.get("code") != "200000":
                raise ExchangeAPIError(
                    message=f"KuCoin API error: {data.get('msg')}",
                    code=data.get("code"),
                    exchange="kucoin"
                )
            
            # Map response to OrderResult
            order_id = data["data"]["orderId"]
            
            return OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.NEW,
                quantity=request.quantity,
                filled_quantity=Decimal("0"),
                price=request.price,
                average_price=None,
                timestamp=int(time.time() * 1000)
            )
        
        except ExchangeAPIError:
            raise
        except Exception as e:
            logger.error(
                f"KuCoin place_order failed: {e}",
                extra={"exchange": "kucoin", "symbol": request.symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to place KuCoin order: {e}",
                exchange="kucoin",
                original_error=e
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """
        Cancel order on KuCoin Futures.
        
        API: DELETE /api/v1/orders/{orderId}
        """
        try:
            endpoint = f"/api/v1/orders/{order_id}"
            headers = self._build_headers("DELETE", endpoint)
            
            url = self.base_url + endpoint
            response = await self.http_client.delete(url, headers=headers)
            
            if response.status_code != 200:
                raise ExchangeAPIError(
                    message=f"KuCoin cancel failed: {response.text}",
                    code=response.status_code,
                    exchange="kucoin"
                )
            
            data = response.json()
            
            success = data.get("code") == "200000"
            
            return CancelResult(
                order_id=order_id,
                symbol=symbol,
                success=success,
                timestamp=int(time.time() * 1000)
            )
        
        except ExchangeAPIError:
            raise
        except Exception as e:
            logger.error(
                f"KuCoin cancel_order failed: {e}",
                extra={"exchange": "kucoin", "order_id": order_id},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to cancel KuCoin order: {e}",
                exchange="kucoin",
                original_error=e
            )
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get open positions from KuCoin Futures.
        
        API: GET /api/v1/positions
        
        NOTE: Implementation follows same pattern as place_order/cancel_order.
        Maps KuCoin position format -> Position model.
        """
        # TODO: Implement position fetching
        raise NotImplementedError("KuCoin get_open_positions - coming soon")
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """
        Get account balances from KuCoin Futures.
        
        API: GET /api/v1/account-overview
        
        NOTE: Implementation follows same pattern as other methods.
        """
        # TODO: Implement balance fetching
        raise NotImplementedError("KuCoin get_balances - coming soon")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """Get klines from KuCoin Futures."""
        raise NotImplementedError("KuCoin get_klines - coming soon")
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """Get order status from KuCoin Futures."""
        raise NotImplementedError("KuCoin get_order_status - coming soon")
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for KuCoin Futures symbol."""
        raise NotImplementedError("KuCoin set_leverage - coming soon")
    
    async def close_position(self, symbol: str) -> OrderResult:
        """Close position on KuCoin Futures."""
        raise NotImplementedError("KuCoin close_position - coming soon")
    
    def get_exchange_name(self) -> str:
        """Get exchange name."""
        return self.exchange_name
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def health(self) -> Dict[str, Any]:
        """
        Check KuCoin API health.
        
        Performs lightweight server time check to verify connectivity.
        
        Returns:
            Dict with status, latency_ms, last_error
        """
        try:
            start = time.time()
            
            endpoint = "/api/v1/timestamp"
            url = self.base_url + endpoint
            
            response = await self.http_client.get(url)
            
            latency_ms = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                return {
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "last_error": None
                }
            else:
                return {
                    "status": "degraded",
                    "latency_ms": latency_ms,
                    "last_error": f"HTTP {response.status_code}"
                }
        
        except Exception as e:
            return {
                "status": "down",
                "latency_ms": None,
                "last_error": str(e)
            }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map OrderType enum to KuCoin order type string."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_MARKET: "market",  # With stopPrice
            OrderType.STOP_LIMIT: "limit",    # With stopPrice
        }
        return mapping.get(order_type, "limit")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        await self.http_client.aclose()

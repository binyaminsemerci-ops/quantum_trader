"""
Firi Exchange Adapter

EPIC-EXCH-003: Firi (Nordic crypto exchange) integration implementing IExchangeClient.
Supports spot trading (market/limit orders) for Norwegian/Danish crypto market.
"""

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


class FiriAdapter:
    """
    Firi adapter implementing IExchangeClient.
    
    Uses Firi Trading API v2 with HMAC-SHA256 authentication.
    Supports spot trading (BTC, ETH, etc. vs NOK/EUR).
    
    Args:
        api_key: Firi API key
        client_id: Firi client ID (user identifier)
        secret_key: Firi secret key for HMAC signing
        testnet: Use sandbox environment (default: False)
    
    Example:
        adapter = FiriAdapter(
            api_key=os.getenv("FIRI_API_KEY"),
            client_id=os.getenv("FIRI_CLIENT_ID"),
            secret_key=os.getenv("FIRI_SECRET_KEY"),
            testnet=False
        )
        result = await adapter.place_order(order_request)
    
    Note:
        Firi is a regulated Nordic exchange (Norway/Denmark).
        API documentation: developers.firi.com
    """
    
    def __init__(
        self,
        api_key: str,
        client_id: str,
        secret_key: str,
        testnet: bool = False
    ):
        self.api_key = api_key
        self.client_id = client_id
        self.secret_key = secret_key
        self.testnet = testnet
        self.exchange_name = "firi"
        
        # Base URLs
        if testnet:
            self.base_url = "https://api-sandbox.firi.com"  # Hypothetical sandbox
        else:
            self.base_url = "https://api.firi.com"
        
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        logger.info(
            f"FiriAdapter initialized (testnet={testnet})",
            extra={"exchange": "firi", "testnet": testnet}
        )
    
    # ========================================================================
    # Authentication & Signing
    # ========================================================================
    
    def _generate_signature(
        self,
        method: str,
        path: str,
        timestamp: str,
        body: str = ""
    ) -> str:
        """
        Generate HMAC-SHA256 signature for Firi API.
        
        Signature format: HMAC-SHA256(method + path + timestamp + body, secret_key)
        Result is hex-encoded string.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API path (e.g., /v2/orders)
            timestamp: Unix timestamp in milliseconds (string)
            body: Request body (JSON string, empty for GET)
        
        Returns:
            Hex-encoded signature
        """
        # Firi signature: method + path + timestamp + body
        message = method + path + timestamp + body
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Build authenticated request headers for Firi API.
        
        Args:
            method: HTTP method
            path: API path
            body: Request body (empty for GET)
        
        Returns:
            Headers dict with signature and authentication
        """
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(method, path, timestamp, body)
        
        return {
            "X-Firi-Client-Id": self.client_id,
            "X-Firi-Api-Key": self.api_key,
            "X-Firi-Signature": signature,
            "X-Firi-Timestamp": timestamp,
            "Content-Type": "application/json"
        }
    
    # ========================================================================
    # IExchangeClient Implementation
    # ========================================================================
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place order on Firi.
        
        Maps OrderRequest -> Firi API format -> OrderResult.
        
        API: POST /v2/orders
        """
        try:
            path = "/v2/orders"
            
            # Map to Firi order format
            order_data = {
                "market": request.symbol,  # e.g., BTCNOK
                "type": self._map_order_type(request.order_type),
                "amount": str(request.quantity),
            }
            
            # Side: "buy" or "sell"
            if request.side == OrderSide.BUY:
                order_data["side"] = "buy"
            else:
                order_data["side"] = "sell"
            
            # Add price for limit orders
            if request.order_type == OrderType.LIMIT and request.price:
                order_data["price"] = str(request.price)
            
            # Client order ID
            if request.client_order_id:
                order_data["clientOrderId"] = request.client_order_id
            
            body = httpx._utils.json.dumps(order_data)
            headers = self._build_headers("POST", path, body)
            
            # Execute request
            url = self.base_url + path
            response = await self.http_client.post(url, headers=headers, content=body)
            
            if response.status_code not in (200, 201):
                raise ExchangeAPIError(
                    message=f"Firi order failed: {response.text}",
                    code=response.status_code,
                    exchange="firi"
                )
            
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                raise ExchangeAPIError(
                    message=f"Firi API error: {data['error']}",
                    code=data.get("code"),
                    exchange="firi"
                )
            
            # Map response to OrderResult
            order_id = data.get("id", "")
            
            return OrderResult(
                order_id=str(order_id),
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=self._map_order_status(data.get("status", "pending")),
                quantity=request.quantity,
                filled_quantity=Decimal(data.get("filled", "0")),
                price=request.price,
                average_price=Decimal(data["price"]) if data.get("price") else None,
                timestamp=int(time.time() * 1000)
            )
        
        except ExchangeAPIError:
            raise
        except Exception as e:
            logger.error(
                f"Firi place_order failed: {e}",
                extra={"exchange": "firi", "symbol": request.symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to place Firi order: {e}",
                exchange="firi",
                original_error=e
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """
        Cancel order on Firi.
        
        API: DELETE /v2/orders/{order_id}
        """
        try:
            path = f"/v2/orders/{order_id}"
            headers = self._build_headers("DELETE", path)
            
            url = self.base_url + path
            response = await self.http_client.delete(url, headers=headers)
            
            if response.status_code not in (200, 204):
                raise ExchangeAPIError(
                    message=f"Firi cancel failed: {response.text}",
                    code=response.status_code,
                    exchange="firi"
                )
            
            # Firi returns 204 No Content on success
            success = response.status_code in (200, 204)
            
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
                f"Firi cancel_order failed: {e}",
                extra={"exchange": "firi", "order_id": order_id},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to cancel Firi order: {e}",
                exchange="firi",
                original_error=e
            )
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get open positions from Firi.
        
        NOTE: Firi is spot-only, so this returns empty list.
        For spot balances, use get_balances().
        
        API: N/A (Firi doesn't support futures/positions)
        """
        # Firi is spot-only, no positions
        logger.debug(
            "Firi get_open_positions called (spot-only exchange, returning empty)",
            extra={"exchange": "firi"}
        )
        return []
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """
        Get account balances from Firi.
        
        API: GET /v2/balances
        
        TODO: Map Firi balance response to Balance model.
        """
        try:
            path = "/v2/balances"
            headers = self._build_headers("GET", path)
            
            url = self.base_url + path
            response = await self.http_client.get(url, headers=headers)
            
            if response.status_code != 200:
                raise ExchangeAPIError(
                    message=f"Firi get_balances failed: {response.text}",
                    code=response.status_code,
                    exchange="firi"
                )
            
            data = response.json()
            
            balances = []
            for item in data.get("balances", []):
                asset_name = item.get("currency", "")
                if asset and asset_name != asset:
                    continue
                
                balances.append(Balance(
                    asset=asset_name,
                    free=Decimal(item.get("available", "0")),
                    locked=Decimal(item.get("reserved", "0")),
                    total=Decimal(item.get("balance", "0"))
                ))
            
            return balances
        
        except ExchangeAPIError:
            raise
        except Exception as e:
            logger.error(
                f"Firi get_balances failed: {e}",
                extra={"exchange": "firi"},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to get Firi balances: {e}",
                exchange="firi",
                original_error=e
            )
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """
        Get klines from Firi.
        
        TODO: Implement if Firi provides historical OHLCV data.
        For now, raise NotImplementedError.
        """
        raise NotImplementedError("Firi get_klines - coming soon")
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """
        Get order status from Firi.
        
        API: GET /v2/orders/{order_id}
        
        TODO: Implement full order status query.
        """
        raise NotImplementedError("Firi get_order_status - coming soon")
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for Firi symbol.
        
        NOTE: Firi is spot-only, no leverage support.
        """
        logger.warning(
            "Firi set_leverage called (spot-only exchange, no-op)",
            extra={"exchange": "firi", "symbol": symbol}
        )
        return False
    
    async def close_position(self, symbol: str) -> OrderResult:
        """
        Close position on Firi.
        
        NOTE: Firi is spot-only, no positions to close.
        """
        raise NotImplementedError("Firi close_position - not applicable for spot exchange")
    
    def get_exchange_name(self) -> str:
        """Get exchange name."""
        return self.exchange_name
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def health(self) -> Dict[str, Any]:
        """
        Check Firi API health.
        
        Performs lightweight server time check to verify connectivity.
        
        Returns:
            Dict with status, latency_ms, last_error
        """
        try:
            start = time.time()
            
            path = "/v2/time"
            url = self.base_url + path
            
            # Public endpoint (no auth required)
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
        """Map OrderType enum to Firi order type string."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
        }
        return mapping.get(order_type, "limit")
    
    def _map_order_status(self, firi_status: str) -> OrderStatus:
        """Map Firi order status to OrderStatus enum."""
        mapping = {
            "pending": OrderStatus.NEW,
            "open": OrderStatus.NEW,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(firi_status.lower(), OrderStatus.NEW)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        await self.http_client.aclose()

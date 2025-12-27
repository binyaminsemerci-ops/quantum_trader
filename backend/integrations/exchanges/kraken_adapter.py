"""
Kraken Exchange Adapter

EPIC-EXCH-002: Kraken Futures integration implementing IExchangeClient.
Supports multi-collateral perpetual futures trading.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
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


class KrakenAdapter:
    """
    Kraken Futures adapter implementing IExchangeClient.
    
    Uses Kraken Futures API v3 with HMAC-SHA512 authentication.
    Supports multi-collateral perpetual contracts.
    
    Args:
        api_key: Kraken API key (public key)
        api_secret: Kraken API secret (base64-encoded private key)
        testnet: Use demo environment (default: False)
    
    Example:
        adapter = KrakenAdapter(
            api_key=os.getenv("KRAKEN_API_KEY"),
            api_secret=os.getenv("KRAKEN_API_SECRET"),
            testnet=False
        )
        result = await adapter.place_order(order_request)
    
    Note:
        Kraken Futures uses base64-encoded API secret for signing.
        Testnet environment is "demo-futures.kraken.com".
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange_name = "kraken"
        
        # Base URLs
        if testnet:
            self.base_url = "https://demo-futures.kraken.com"
        else:
            self.base_url = "https://futures.kraken.com"
        
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        logger.info(
            f"KrakenAdapter initialized (testnet={testnet})",
            extra={"exchange": "kraken", "testnet": testnet}
        )
    
    # ========================================================================
    # Authentication & Signing
    # ========================================================================
    
    def _generate_signature(
        self,
        endpoint: str,
        nonce: str,
        postdata: str
    ) -> str:
        """
        Generate HMAC-SHA512 signature for Kraken Futures API.
        
        Signature format:
        1. Concatenate endpoint path + nonce + postdata
        2. Hash with SHA256
        3. Sign with HMAC-SHA512 using decoded api_secret
        4. Base64 encode result
        
        Args:
            endpoint: API endpoint path (e.g., /derivatives/api/v3/sendorder)
            nonce: Unique nonce (timestamp in milliseconds)
            postdata: URL-encoded POST data
        
        Returns:
            Base64-encoded signature
        """
        # Step 1: Concatenate endpoint + nonce + postdata
        message = endpoint + nonce + postdata
        
        # Step 2: SHA256 hash
        sha256_hash = hashlib.sha256(message.encode('utf-8')).digest()
        
        # Step 3: HMAC-SHA512 with base64-decoded secret
        api_secret_decoded = base64.b64decode(self.api_secret)
        signature = hmac.new(
            api_secret_decoded,
            sha256_hash,
            hashlib.sha512
        ).digest()
        
        # Step 4: Base64 encode
        return base64.b64encode(signature).decode('utf-8')
    
    def _build_headers(
        self,
        endpoint: str,
        postdata: str = ""
    ) -> Dict[str, str]:
        """
        Build authenticated request headers for Kraken Futures API.
        
        Args:
            endpoint: API endpoint path
            postdata: URL-encoded POST data (empty for GET)
        
        Returns:
            Headers dict with signature and authentication
        """
        nonce = str(int(time.time() * 1000))
        signature = self._generate_signature(endpoint, nonce, postdata)
        
        return {
            "APIKey": self.api_key,
            "Authent": signature,
            "Nonce": nonce,
            "Content-Type": "application/x-www-form-urlencoded"
        }
    
    # ========================================================================
    # IExchangeClient Implementation
    # ========================================================================
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place order on Kraken Futures.
        
        Maps OrderRequest -> Kraken API format -> OrderResult.
        
        API: POST /derivatives/api/v3/sendorder
        """
        try:
            endpoint = "/derivatives/api/v3/sendorder"
            
            # Map to Kraken order format
            order_params = {
                "symbol": request.symbol,  # e.g., PI_XBTUSD
                "side": "buy" if request.side == OrderSide.BUY else "sell",
                "orderType": self._map_order_type(request.order_type),
                "size": int(request.quantity),  # Kraken uses contracts (integer)
            }
            
            # Add price for limit orders
            if request.order_type == OrderType.LIMIT and request.price:
                order_params["limitPrice"] = float(request.price)
            
            # Add stop price for stop orders
            if request.stop_price:
                order_params["stopPrice"] = float(request.stop_price)
            
            # Reduce-only flag
            if request.reduce_only:
                order_params["reduceOnly"] = "true"
            
            # Client order ID
            if request.client_order_id:
                order_params["cliOrdId"] = request.client_order_id
            
            # URL-encode POST data
            postdata = urllib.parse.urlencode(order_params)
            headers = self._build_headers(endpoint, postdata)
            
            # Execute request
            url = self.base_url + endpoint
            response = await self.http_client.post(
                url,
                headers=headers,
                content=postdata
            )
            
            if response.status_code != 200:
                raise ExchangeAPIError(
                    message=f"Kraken order failed: {response.text}",
                    code=response.status_code,
                    exchange="kraken"
                )
            
            data = response.json()
            
            # Check for API errors
            if data.get("result") != "success":
                error_msg = data.get("error", "Unknown error")
                raise ExchangeAPIError(
                    message=f"Kraken API error: {error_msg}",
                    code=data.get("errorCode"),
                    exchange="kraken"
                )
            
            # Map response to OrderResult
            send_status = data.get("sendStatus", {})
            order_id = send_status.get("order_id", "")
            
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
                f"Kraken place_order failed: {e}",
                extra={"exchange": "kraken", "symbol": request.symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to place Kraken order: {e}",
                exchange="kraken",
                original_error=e
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """
        Cancel order on Kraken Futures.
        
        API: POST /derivatives/api/v3/cancelorder
        """
        try:
            endpoint = "/derivatives/api/v3/cancelorder"
            
            # Build cancel request
            cancel_params = {
                "order_id": order_id
            }
            
            postdata = urllib.parse.urlencode(cancel_params)
            headers = self._build_headers(endpoint, postdata)
            
            url = self.base_url + endpoint
            response = await self.http_client.post(
                url,
                headers=headers,
                content=postdata
            )
            
            if response.status_code != 200:
                raise ExchangeAPIError(
                    message=f"Kraken cancel failed: {response.text}",
                    code=response.status_code,
                    exchange="kraken"
                )
            
            data = response.json()
            
            success = data.get("result") == "success"
            
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
                f"Kraken cancel_order failed: {e}",
                extra={"exchange": "kraken", "order_id": order_id},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Failed to cancel Kraken order: {e}",
                exchange="kraken",
                original_error=e
            )
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get open positions from Kraken Futures.
        
        API: GET /derivatives/api/v3/openpositions
        
        NOTE: Implementation follows same pattern as place_order/cancel_order.
        Maps Kraken position format -> Position model.
        """
        # TODO: Implement position fetching
        raise NotImplementedError("Kraken get_open_positions - coming soon")
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """
        Get account balances from Kraken Futures.
        
        API: GET /derivatives/api/v3/accounts
        
        NOTE: Implementation follows same pattern as other methods.
        """
        # TODO: Implement balance fetching
        raise NotImplementedError("Kraken get_balances - coming soon")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """Get klines from Kraken Futures."""
        raise NotImplementedError("Kraken get_klines - coming soon")
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """Get order status from Kraken Futures."""
        raise NotImplementedError("Kraken get_order_status - coming soon")
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for Kraken Futures symbol.
        
        NOTE: Kraken Futures uses margin requirement instead of leverage.
        This method would convert leverage -> margin percentage.
        """
        raise NotImplementedError("Kraken set_leverage - coming soon")
    
    async def close_position(self, symbol: str) -> OrderResult:
        """Close position on Kraken Futures."""
        raise NotImplementedError("Kraken close_position - coming soon")
    
    def get_exchange_name(self) -> str:
        """Get exchange name."""
        return self.exchange_name
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def health(self) -> Dict[str, Any]:
        """
        Check Kraken API health.
        
        Performs lightweight server time check to verify connectivity.
        
        Returns:
            Dict with status, latency_ms, last_error
        """
        try:
            start = time.time()
            
            # Public endpoint (no auth required)
            endpoint = "/derivatives/api/v3/instruments"
            url = self.base_url + endpoint
            
            response = await self.http_client.get(url)
            
            latency_ms = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("result") == "success":
                    return {
                        "status": "ok",
                        "latency_ms": latency_ms,
                        "last_error": None
                    }
                else:
                    return {
                        "status": "degraded",
                        "latency_ms": latency_ms,
                        "last_error": data.get("error", "Unknown error")
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
        """Map OrderType enum to Kraken order type string."""
        mapping = {
            OrderType.MARKET: "mkt",
            OrderType.LIMIT: "lmt",
            OrderType.STOP_MARKET: "stp",
            OrderType.STOP_LIMIT: "stop_limit",
        }
        return mapping.get(order_type, "lmt")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        await self.http_client.aclose()

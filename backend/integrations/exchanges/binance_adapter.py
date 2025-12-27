"""
Binance Exchange Adapter

EPIC-EXCH-001: Real implementation of IExchangeClient for Binance Futures.
Wraps existing Binance client with unified interface.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any

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


class BinanceAdapter:
    """
    Binance Futures adapter implementing IExchangeClient.
    
    Wraps python-binance Client with rate limiting and error handling.
    Maps generic models <-> Binance API format.
    
    Args:
        client: Binance Client instance (from python-binance)
        wrapper: Optional BinanceClientWrapper for rate limiting
        testnet: Whether using testnet (default: False)
    
    Example:
        from binance.client import Client
        from backend.integrations.binance import create_binance_wrapper
        
        client = Client(api_key, api_secret)
        wrapper = create_binance_wrapper()
        adapter = BinanceAdapter(client, wrapper)
        
        result = await adapter.place_order(order_request)
    """
    
    def __init__(self, client, wrapper=None, testnet: bool = False):
        self.client = client
        self.wrapper = wrapper  # Optional rate limiter wrapper
        self.testnet = testnet
        self.exchange_name = "binance"
        
        # Log base URL to verify testnet config
        base_url = getattr(client, 'API_URL', 'unknown')
        logger.info(
            f"BinanceAdapter initialized (testnet={testnet}, base_url={base_url})",
            extra={"exchange": "binance", "testnet": testnet, "base_url": base_url}
        )
    
    # ========================================================================
    # IExchangeClient Implementation
    # ========================================================================
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place order on Binance Futures.
        
        Maps OrderRequest -> Binance API format -> OrderResult.
        """
        try:
            # Build Binance API parameters
            params = self._build_order_params(request)
            
            # Execute order via wrapper (if available) or directly
            if self.wrapper:
                response = await self.wrapper.call_async(
                    self.client.futures_create_order,
                    **params
                )
            else:
                response = await asyncio.to_thread(
                    self.client.futures_create_order,
                    **params
                )
            
            # Map response to OrderResult
            result = self._map_order_response(response, request)
            
            logger.info(
                f"Order placed: {result.order_id} ({request.symbol} {request.side} {request.quantity})",
                extra={
                    "exchange": "binance",
                    "order_id": result.order_id,
                    "symbol": request.symbol,
                    "side": request.side.value,
                    "type": request.order_type.value,
                    "status": result.status.value,
                }
            )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Order placement failed: {e}",
                extra={
                    "exchange": "binance",
                    "symbol": request.symbol,
                    "error": str(e),
                },
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance order placement failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> CancelResult:
        """Cancel order on Binance Futures."""
        try:
            if self.wrapper:
                response = await self.wrapper.call_async(
                    self.client.futures_cancel_order,
                    symbol=symbol.upper(),
                    orderId=int(order_id)
                )
            else:
                response = await asyncio.to_thread(
                    self.client.futures_cancel_order,
                    symbol=symbol.upper(),
                    orderId=int(order_id)
                )
            
            result = CancelResult(
                order_id=str(response['orderId']),
                symbol=response['symbol'],
                status=self._map_order_status(response['status']),
                success=response['status'] == 'CANCELED',
                exchange="binance"
            )
            
            logger.info(
                f"Order canceled: {order_id}",
                extra={"exchange": "binance", "order_id": order_id, "symbol": symbol}
            )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Order cancellation failed: {e}",
                extra={"exchange": "binance", "order_id": order_id, "symbol": symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance order cancellation failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions from Binance Futures."""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol.upper()
            
            if self.wrapper:
                response = await self.wrapper.call_async(
                    self.client.futures_position_information,
                    **params
                )
            else:
                response = await asyncio.to_thread(
                    self.client.futures_position_information,
                    **params
                )
            
            # Filter to actual positions (non-zero quantity)
            positions = [
                self._map_position(pos)
                for pos in response
                if float(pos.get('positionAmt', 0)) != 0
            ]
            
            logger.debug(
                f"Fetched {len(positions)} open positions",
                extra={"exchange": "binance", "symbol": symbol, "count": len(positions)}
            )
            
            return positions
        
        except Exception as e:
            logger.error(
                f"Position fetch failed: {e}",
                extra={"exchange": "binance", "symbol": symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance position fetch failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    async def get_balances(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balances from Binance Futures."""
        try:
            if self.wrapper:
                response = await self.wrapper.call_async(
                    self.client.futures_account_balance
                )
            else:
                response = await asyncio.to_thread(
                    self.client.futures_account_balance
                )
            
            balances = [
                self._map_balance(bal)
                for bal in response
                if not asset or bal['asset'].upper() == asset.upper()
            ]
            
            # Filter to non-zero balances
            balances = [b for b in balances if b.total > 0]
            
            logger.debug(
                f"Fetched {len(balances)} balances",
                extra={"exchange": "binance", "asset": asset, "count": len(balances)}
            )
            
            return balances
        
        except Exception as e:
            logger.error(
                f"Balance fetch failed: {e}",
                extra={"exchange": "binance", "asset": asset},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance balance fetch failed: {e}",
                exchange="binance",
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
        """Get candlestick data from Binance."""
        try:
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            if self.wrapper:
                response = await self.wrapper.call_async(
                    self.client.futures_klines,
                    **params
                )
            else:
                response = await asyncio.to_thread(
                    self.client.futures_klines,
                    **params
                )
            
            klines = [self._map_kline(k, symbol, interval) for k in response]
            
            logger.debug(
                f"Fetched {len(klines)} klines",
                extra={
                    "exchange": "binance",
                    "symbol": symbol,
                    "interval": interval,
                    "count": len(klines)
                }
            )
            
            return klines
        
        except Exception as e:
            logger.error(
                f"Kline fetch failed: {e}",
                extra={"exchange": "binance", "symbol": symbol, "interval": interval},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance kline fetch failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """Get order status from Binance."""
        try:
            if self.wrapper:
                response = await self.wrapper.call_async(
                    self.client.futures_get_order,
                    symbol=symbol.upper(),
                    orderId=int(order_id)
                )
            else:
                response = await asyncio.to_thread(
                    self.client.futures_get_order,
                    symbol=symbol.upper(),
                    orderId=int(order_id)
                )
            
            return self._map_order_response(response, None)
        
        except Exception as e:
            logger.error(
                f"Order status fetch failed: {e}",
                extra={"exchange": "binance", "order_id": order_id, "symbol": symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance order status fetch failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol on Binance Futures."""
        try:
            if self.wrapper:
                await self.wrapper.call_async(
                    self.client.futures_change_leverage,
                    symbol=symbol.upper(),
                    leverage=leverage
                )
            else:
                await asyncio.to_thread(
                    self.client.futures_change_leverage,
                    symbol=symbol.upper(),
                    leverage=leverage
                )
            
            logger.info(
                f"Leverage set: {symbol} -> {leverage}x",
                extra={"exchange": "binance", "symbol": symbol, "leverage": leverage}
            )
            
            return True
        
        except Exception as e:
            logger.error(
                f"Leverage change failed: {e}",
                extra={"exchange": "binance", "symbol": symbol, "leverage": leverage},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance leverage change failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    async def close_position(self, symbol: str) -> OrderResult:
        """Close position by placing market order in opposite direction."""
        try:
            # Get current position
            positions = await self.get_open_positions(symbol=symbol)
            if not positions:
                raise ExchangeAPIError(
                    message=f"No open position for {symbol}",
                    exchange="binance"
                )
            
            position = positions[0]
            
            # Create close order (opposite side, reduce-only)
            close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            close_request = OrderRequest(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reduce_only=True
            )
            
            result = await self.place_order(close_request)
            
            logger.info(
                f"Position closed: {symbol} ({position.quantity})",
                extra={
                    "exchange": "binance",
                    "symbol": symbol,
                    "quantity": str(position.quantity),
                    "side": close_side.value
                }
            )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Position close failed: {e}",
                extra={"exchange": "binance", "symbol": symbol},
                exc_info=True
            )
            raise ExchangeAPIError(
                message=f"Binance position close failed: {e}",
                exchange="binance",
                original_error=e
            )
    
    def get_exchange_name(self) -> str:
        """Return exchange name."""
        return self.exchange_name
    
    # ========================================================================
    # Mapping Helpers (Binance <-> Unified Models)
    # ========================================================================
    
    def _build_order_params(self, request: OrderRequest) -> Dict[str, Any]:
        """Convert OrderRequest to Binance API parameters."""
        params = {
            'symbol': request.symbol.upper(),
            'side': request.side.value,
            'type': request.order_type.value,
            'quantity': float(request.quantity),
        }
        
        if request.price:
            params['price'] = float(request.price)
        
        if request.stop_price:
            params['stopPrice'] = float(request.stop_price)
        
        if request.time_in_force:
            params['timeInForce'] = request.time_in_force.value
        
        if request.reduce_only:
            params['reduceOnly'] = True
        
        if request.position_side:
            params['positionSide'] = request.position_side.value
        
        if request.client_order_id:
            params['newClientOrderId'] = request.client_order_id
        
        return params
    
    def _map_order_response(self, response: Dict, request: Optional[OrderRequest]) -> OrderResult:
        """Map Binance order response to OrderResult."""
        return OrderResult(
            order_id=str(response['orderId']),
            client_order_id=response.get('clientOrderId'),
            symbol=response['symbol'],
            side=OrderSide(response['side']),
            order_type=OrderType(response['type']),
            quantity=Decimal(str(response.get('origQty', 0))),
            filled_quantity=Decimal(str(response.get('executedQty', 0))),
            price=Decimal(str(response['price'])) if response.get('price') else None,
            average_price=Decimal(str(response.get('avgPrice', 0))) if response.get('avgPrice') else None,
            status=self._map_order_status(response['status']),
            timestamp=datetime.fromtimestamp(response['updateTime'] / 1000),
            exchange="binance",
            raw_response=response
        )
    
    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Binance order status to OrderStatus enum."""
        mapping = {
            'NEW': OrderStatus.NEW,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.NEW)
    
    def _map_position(self, pos: Dict) -> Position:
        """Map Binance position to Position model."""
        position_amt = float(pos['positionAmt'])
        side = OrderSide.BUY if position_amt > 0 else OrderSide.SELL
        
        # Handle optional fields with safe defaults
        leverage = int(pos.get('leverage', 1))
        liquidation_price = pos.get('liquidationPrice')
        if liquidation_price and liquidation_price != "0":
            liq_price = Decimal(str(liquidation_price))
        else:
            liq_price = None
        
        # Calculate margin: for cross margin, use notional / leverage
        isolated_margin = Decimal(str(pos.get('isolatedMargin', 0)))
        if isolated_margin > 0:
            margin = isolated_margin
        else:
            # Cross margin: calculate from notional value
            notional = Decimal(str(abs(position_amt))) * Decimal(str(pos['markPrice']))
            margin = notional / Decimal(str(leverage)) if leverage > 0 else notional
        
        return Position(
            symbol=pos['symbol'],
            side=side,
            quantity=Decimal(str(abs(position_amt))),
            entry_price=Decimal(str(pos['entryPrice'])),
            mark_price=Decimal(str(pos['markPrice'])),
            liquidation_price=liq_price,
            unrealized_pnl=Decimal(str(pos['unRealizedProfit'])),
            leverage=leverage,
            margin=margin,
            exchange="binance",
            position_side=PositionSide(pos.get('positionSide', 'BOTH'))
        )
    
    def _map_balance(self, bal: Dict) -> Balance:
        """Map Binance balance to Balance model."""
        available = Decimal(str(bal['availableBalance']))
        balance = Decimal(str(bal['balance']))
        
        # Handle negative locked balance (can happen on Binance)
        locked = balance - available
        if locked < 0:
            locked = Decimal("0")
        
        return Balance(
            asset=bal['asset'],
            free=available,
            locked=locked,
            total=balance,
            exchange="binance"
        )
    
    def _map_kline(self, k: List, symbol: str, interval: str) -> Kline:
        """Map Binance kline to Kline model."""
        return Kline(
            symbol=symbol,
            interval=interval,
            open_time=datetime.fromtimestamp(k[0] / 1000),
            close_time=datetime.fromtimestamp(k[6] / 1000),
            open=Decimal(str(k[1])),
            high=Decimal(str(k[2])),
            low=Decimal(str(k[3])),
            close=Decimal(str(k[4])),
            volume=Decimal(str(k[5])),
            quote_volume=Decimal(str(k[7])),
            trades=int(k[8]),
            exchange="binance"
        )

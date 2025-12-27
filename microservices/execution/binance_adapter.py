"""
Execution Service - Binance Client Adapter

Clean Binance integration without monolith dependencies.
Supports: Testnet, Mainnet, Paper Trading.
"""
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum

try:
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logging.warning("python-binance not installed - only PAPER mode available")


logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    TESTNET = "TESTNET"
    LIVE = "LIVE"


class BinanceAdapter:
    """
    Binance API adapter for order execution.
    
    Modes:
    - PAPER: Simulated execution (no API calls)
    - TESTNET: Binance futures testnet
    - LIVE: Real trading (use with extreme caution)
    """
    
    def __init__(
        self,
        mode: ExecutionMode,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.mode = mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.client: Optional[AsyncClient] = None
        
        # Paper trading state
        self.paper_order_id = 1000
        self.paper_orders: Dict[str, Dict] = {}
        
        logger.info(f"[BINANCE-ADAPTER] Initialized in {mode} mode")
    
    async def connect(self):
        """Initialize Binance client"""
        if self.mode == ExecutionMode.PAPER:
            logger.info("[BINANCE-ADAPTER] Paper mode - no API connection")
            return
        
        if not BINANCE_AVAILABLE:
            raise RuntimeError("python-binance not installed - cannot use TESTNET/LIVE mode")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials required for TESTNET/LIVE mode")
        
        # Create Binance Futures client
        testnet = (self.mode == ExecutionMode.TESTNET)
        
        if testnet:
            # Binance Futures Testnet Configuration
            # CRITICAL: python-binance testnet=True uses WRONG URL for futures testnet!
            # testnet=True gives: https://fapi.binance.com/fapi (LIVE!)
            # We need: https://testnet.binancefuture.com/fapi (TESTNET!)
            logger.info(f"[BINANCE-ADAPTER] API Key (first 10 chars): {self.api_key[:10]}...")
            logger.info(f"[BINANCE-ADAPTER] API Secret (first 10 chars): {self.api_secret[:10]}...")
            
            # Create client WITHOUT testnet=True to avoid wrong URL
            self.client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Manually override ALL base URLs to futures testnet BEFORE any API calls
            self.client.API_URL = 'https://testnet.binancefuture.com'
            self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
            self.client.FUTURES_DATA_URL = 'https://testnet.binancefuture.com/fapi'
            
            logger.info(f"[BINANCE-ADAPTER] Using Binance Futures TESTNET")
            logger.info(f"[BINANCE-ADAPTER] API_URL: {self.client.API_URL}")
            logger.info(f"[BINANCE-ADAPTER] FUTURES_URL: {self.client.FUTURES_URL}")
        else:
            self.client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        
        # Test connection and set position mode
        try:
            account = self.client.futures_account()
            balance = float(account['totalWalletBalance'])
            
            # Set hedge mode to false (one-way mode) for testnet
            try:
                self.client.futures_change_position_mode(dualSidePosition='false')
                logger.info("[BINANCE-ADAPTER] Position mode set to One-way")
            except Exception as pos_error:
                logger.warning(f"[BINANCE-ADAPTER] Position mode warning: {pos_error}")
            
            logger.info(
                f"[BINANCE-ADAPTER] Connected to {'TESTNET' if testnet else 'MAINNET'}, "
                f"balance: ${balance:.2f}"
            )
        except Exception as e:
            logger.error(f"[BINANCE-ADAPTER] Connection test failed: {e}")
            raise
    
    async def close(self):
        """Close Binance client"""
        if self.client:
            # UMFutures doesn't need explicit close
            self.client = None
            logger.info("[BINANCE-ADAPTER] Disconnected")
    
    def _get_quantity_precision(self, symbol: str) -> int:
        """
        Get quantity precision for a symbol from exchange info.
        
        Returns:
            int: Number of decimal places allowed (default 3 if not found)
        """
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    # Get quantityPrecision from filters
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            # Count decimal places in step_size
                            # e.g., 0.001 -> 3, 0.01 -> 2, 0.1 -> 1, 1.0 -> 0
                            precision = len(str(step_size).rstrip('0').split('.')[-1]) if '.' in str(step_size) else 0
                            logger.debug(f"[BINANCE-ADAPTER] {symbol} quantity precision: {precision}")
                            return precision
            logger.warning(f"[BINANCE-ADAPTER] Could not find precision for {symbol}, using default 3")
            return 3
        except Exception as e:
            logger.error(f"[BINANCE-ADAPTER] Error getting precision for {symbol}: {e}, using default 3")
            return 3
    
    async def place_market_order(
        self,
        symbol: str,
        side: str,  # BUY/SELL
        quantity: float,
        leverage: int = 1
    ) -> Dict[str, Any]:
        """
        Place market order.
        
        Returns:
            {
                "order_id": str,
                "symbol": str,
                "side": str,
                "quantity": float,
                "price": float,
                "status": str,  # FILLED, REJECTED, etc
                "timestamp": str
            }
        """
        if self.mode == ExecutionMode.PAPER:
            return await self._paper_market_order(symbol, side, quantity, leverage)
        
        # Real Binance API call
        try:
            # Round quantity to correct precision for this symbol
            precision = self._get_quantity_precision(symbol)
            quantity = round(quantity, precision)
            logger.info(f"[BINANCE-ADAPTER] Rounded quantity to {precision} decimals: {quantity}")
            
            # Set leverage first
            if leverage > 1:
                self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=leverage
                )
            
            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            
            result = {
                "order_id": str(order['orderId']),
                "symbol": order['symbol'],
                "side": order['side'],
                "quantity": float(order['origQty']),
                "price": float(order.get('avgPrice', 0)) or float(order.get('price', 0)),
                "status": order['status'],
                "timestamp": datetime.utcnow().isoformat(),
                "mode": self.mode.value
            }
            
            logger.info(
                f"[BINANCE-ADAPTER] Order executed: {symbol} {side} {quantity} @ {result['price']}"
            )
            
            return result
            
        except BinanceAPIException as e:
            logger.error(f"[BINANCE-ADAPTER] Order failed: {e}")
            return {
                "order_id": None,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": None,
                "status": "REJECTED",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "mode": self.mode.value
            }
    
    async def _paper_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int
    ) -> Dict[str, Any]:
        """Simulated market order execution"""
        order_id = f"PAPER_{self.paper_order_id}"
        self.paper_order_id += 1
        
        # Simulate price (would need price feed in real implementation)
        # For now, use placeholder
        simulated_price = 50000.0 if "BTC" in symbol else 3000.0
        
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": simulated_price,
            "status": "FILLED",
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "PAPER",
            "leverage": leverage
        }
        
        self.paper_orders[order_id] = order
        
        logger.info(
            f"[BINANCE-ADAPTER] PAPER ORDER: {symbol} {side} {quantity} @ {simulated_price} "
            f"(leverage: {leverage}x)"
        )
        
        return order
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        if self.mode == ExecutionMode.PAPER:
            # Return placeholder
            return 50000.0 if "BTC" in symbol else 3000.0
        
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"[BINANCE-ADAPTER] Failed to get price for {symbol}: {e}")
            return None
    
    async def get_account_balance(self) -> Optional[float]:
        """Get total account balance in USDT"""
        if self.mode == ExecutionMode.PAPER:
            return 10000.0  # Paper account starts with $10k
        
        try:
            account = self.client.futures_account()
            return float(account['totalWalletBalance'])
        except Exception as e:
            logger.error(f"[BINANCE-ADAPTER] Failed to get balance: {e}")
            return None
    
    async def place_stop_loss(
        self,
        symbol: str,
        side: str,  # SELL for long positions, BUY for short positions
        quantity: float,
        stop_price: float
    ) -> Dict[str, Any]:
        """
        Place a STOP_MARKET order (stop loss).
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: SELL (for closing long) or BUY (for closing short)
            quantity: Amount to close
            stop_price: Price that triggers the stop loss
        
        Returns:
            Order result dict
        """
        if self.mode == ExecutionMode.PAPER:
            order_id = f"PAPER_SL_{self.paper_order_id}"
            self.paper_order_id += 1
            return {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "quantity": quantity,
                "stop_price": stop_price,
                "status": "NEW",
                "timestamp": datetime.utcnow().isoformat(),
                "mode": "PAPER"
            }
        
        try:
            # Round quantity to correct precision
            precision = self._get_quantity_precision(symbol)
            quantity = round(quantity, precision)
            
            # Place STOP_MARKET order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                quantity=quantity,
                stopPrice=stop_price
            )
            
            result = {
                "order_id": str(order['orderId']),
                "symbol": order['symbol'],
                "side": order['side'],
                "type": "STOP_MARKET",
                "quantity": float(order['origQty']),
                "stop_price": float(order['stopPrice']),
                "status": order['status'],
                "timestamp": datetime.utcnow().isoformat(),
                "mode": self.mode.value
            }
            
            logger.info(
                f"[BINANCE-ADAPTER] Stop loss placed: {symbol} {side} {quantity} @ stop={stop_price}"
            )
            
            return result
            
        except BinanceAPIException as e:
            logger.error(f"[BINANCE-ADAPTER] Stop loss placement failed: {e}")
            return {
                "order_id": None,
                "symbol": symbol,
                "status": "REJECTED",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def place_take_profit(
        self,
        symbol: str,
        side: str,  # SELL for long positions, BUY for short positions
        quantity: float,
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Place a TAKE_PROFIT_MARKET order.
        
        Args:
            symbol: Trading pair
            side: SELL (for closing long) or BUY (for closing short)
            quantity: Amount to close
            limit_price: Price that triggers the take profit
        
        Returns:
            Order result dict
        """
        if self.mode == ExecutionMode.PAPER:
            order_id = f"PAPER_TP_{self.paper_order_id}"
            self.paper_order_id += 1
            return {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": quantity,
                "limit_price": limit_price,
                "status": "NEW",
                "timestamp": datetime.utcnow().isoformat(),
                "mode": "PAPER"
            }
        
        try:
            # Round quantity to correct precision
            precision = self._get_quantity_precision(symbol)
            quantity = round(quantity, precision)
            
            # Place TAKE_PROFIT_MARKET order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=limit_price  # For TAKE_PROFIT_MARKET, use stopPrice
            )
            
            result = {
                "order_id": str(order['orderId']),
                "symbol": order['symbol'],
                "side": order['side'],
                "type": "TAKE_PROFIT_MARKET",
                "quantity": float(order['origQty']),
                "limit_price": float(order['stopPrice']),
                "status": order['status'],
                "timestamp": datetime.utcnow().isoformat(),
                "mode": self.mode.value
            }
            
            logger.info(
                f"[BINANCE-ADAPTER] Take profit placed: {symbol} {side} {quantity} @ trigger={limit_price}"
            )
            
            return result
            
        except BinanceAPIException as e:
            logger.error(f"[BINANCE-ADAPTER] Take profit placement failed: {e}")
            return {
                "order_id": None,
                "symbol": symbol,
                "status": "REJECTED",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def place_exit_orders(
        self,
        symbol: str,
        side: str,  # LONG or SHORT (position side)
        quantity: float,
        stop_loss_price: float,
        take_profit_prices: list[float],
        take_profit_quantities: list[float]
    ) -> Dict[str, Any]:
        """
        Place multiple exit orders (1 SL + multiple TPs) for a position.
        
        This is the EXIT BRAIN V3 integration point!
        
        Args:
            symbol: Trading pair
            side: Position side (LONG or SHORT)
            quantity: Total position size
            stop_loss_price: Stop loss trigger price
            take_profit_prices: List of TP trigger prices [TP1, TP2, TP3, ...]
            take_profit_quantities: List of TP quantities [Q1, Q2, Q3, ...]
        
        Returns:
            {
                "stop_loss": {...},
                "take_profits": [{...}, {...}, ...],
                "status": "SUCCESS" or "PARTIAL" or "FAILED"
            }
        """
        results = {
            "stop_loss": None,
            "take_profits": [],
            "status": "FAILED"
        }
        
        # Determine order side (opposite of position side)
        order_side = "SELL" if side.upper() in ["LONG", "BUY"] else "BUY"
        
        # 1. Place stop loss (for full position)
        try:
            sl_result = await self.place_stop_loss(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                stop_price=stop_loss_price
            )
            results["stop_loss"] = sl_result
            
            if sl_result["status"] == "REJECTED":
                logger.error(f"[EXIT-ORDERS] Stop loss rejected for {symbol}")
                return results
                
        except Exception as e:
            logger.error(f"[EXIT-ORDERS] Stop loss placement failed: {e}")
            return results
        
        # 2. Place take profit orders (partial closes)
        tp_success_count = 0
        for i, (tp_price, tp_qty) in enumerate(zip(take_profit_prices, take_profit_quantities)):
            try:
                tp_result = await self.place_take_profit(
                    symbol=symbol,
                    side=order_side,
                    quantity=tp_qty,
                    limit_price=tp_price
                )
                results["take_profits"].append(tp_result)
                
                if tp_result["status"] != "REJECTED":
                    tp_success_count += 1
                    
            except Exception as e:
                logger.error(f"[EXIT-ORDERS] TP{i+1} placement failed: {e}")
                results["take_profits"].append({
                    "status": "ERROR",
                    "error": str(e),
                    "level": i+1
                })
        
        # Determine overall status
        if results["stop_loss"]["status"] != "REJECTED" and tp_success_count == len(take_profit_prices):
            results["status"] = "SUCCESS"
            logger.info(
                f"[EXIT-ORDERS] ✅ All exit orders placed for {symbol}: "
                f"1 SL + {tp_success_count} TPs"
            )
        elif tp_success_count > 0:
            results["status"] = "PARTIAL"
            logger.warning(
                f"[EXIT-ORDERS] ⚠️ Partial success for {symbol}: "
                f"{tp_success_count}/{len(take_profit_prices)} TPs placed"
            )
        else:
            logger.error(f"[EXIT-ORDERS] ❌ Failed to place exit orders for {symbol}")
        
        return results

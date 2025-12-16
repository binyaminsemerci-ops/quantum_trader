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
    from binance.client import AsyncClient
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
        
        testnet = (self.mode == ExecutionMode.TESTNET)
        
        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=testnet
        )
        
        # Test connection
        try:
            account = await self.client.futures_account()
            balance = float(account['totalWalletBalance'])
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
            await self.client.close_connection()
            logger.info("[BINANCE-ADAPTER] Disconnected")
    
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
            # Set leverage first
            if leverage > 1:
                await self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=leverage
                )
            
            # Place order
            order = await self.client.futures_create_order(
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
            ticker = await self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"[BINANCE-ADAPTER] Failed to get price for {symbol}: {e}")
            return None
    
    async def get_account_balance(self) -> Optional[float]:
        """Get total account balance in USDT"""
        if self.mode == ExecutionMode.PAPER:
            return 10000.0  # Paper account starts with $10k
        
        try:
            account = await self.client.futures_account()
            return float(account['totalWalletBalance'])
        except Exception as e:
            logger.error(f"[BINANCE-ADAPTER] Failed to get balance: {e}")
            return None

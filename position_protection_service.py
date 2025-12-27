"""
Automatic Position Protection Service
=====================================
Monitors ALL Binance positions and ensures correct TP/SL orders.
Works for both manual and automated trades.

Features:
- Detects positions without TP/SL
- Detects positions with WRONG TP/SL direction
- Calculates ATR-based levels automatically
- Places multi-target orders (TP1, TP2, SL)
- Runs continuously every 60 seconds
- Logs all actions for audit

Author: Quantum Trader Team
Date: 2025-11-26
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from binance.client import Client as BinanceClient
import ccxt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionProtectionService:
    """
    Automatic TP/SL protection for all positions.
    """
    
    def __init__(self, check_interval: int = 60, testnet: bool = False):
        """
        Initialize protection service.
        
        Args:
            check_interval: Seconds between checks (default 60)
            testnet: Use testnet credentials (default False)
        """
        self.check_interval = check_interval
        self.testnet = testnet
        
        # Initialize Binance client
        if testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")
            self.api_url = 'https://testnet.binancefuture.com'
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            self.api_url = 'https://fapi.binance.com'
        
        if not api_key or not api_secret:
            raise ValueError("Missing Binance API credentials!")
        
        self.client = BinanceClient(api_key, api_secret)
        if testnet:
            self.client.API_URL = self.api_url
        
        # Initialize CCXT for ATR calculation
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        logger.info(f"🛡️  Position Protection Service initialized ({'TESTNET' if testnet else 'LIVE'})")
        logger.info(f"📊 Check interval: {check_interval}s")
    
    def calculate_atr(self, symbol: str, period: int = 14, timeframe: str = '15m') -> Optional[float]:
        """
        Calculate ATR for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'DASHUSDT')
            period: ATR period (default 14)
            timeframe: Timeframe (default '15m')
        
        Returns:
            ATR value or None if error
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=period + 1)
            
            if len(ohlcv) < period + 1:
                logger.warning(f"Not enough data for {symbol} ATR calculation")
                return None
            
            # Calculate True Range for each candle
            tr_list = []
            for i in range(1, len(ohlcv)):
                high = ohlcv[i][2]
                low = ohlcv[i][3]
                prev_close = ohlcv[i-1][4]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_list.append(tr)
            
            # Calculate ATR (simple moving average of TR)
            atr = sum(tr_list[-period:]) / period
            
            return atr
        
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None
    
    def get_symbol_precision(self, symbol: str) -> int:
        """Get price precision for symbol."""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if symbol_info:
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    if tick_size >= 1:
                        return 0
                    elif '.' in str(tick_size):
                        return len(str(tick_size).rstrip('0').split('.')[-1])
            
            return 2  # Default
        except Exception as e:
            logger.warning(f"Could not get precision for {symbol}: {e}")
            return 2
    
    def calculate_tpsl_levels(
        self, 
        symbol: str, 
        entry_price: float, 
        side: str,
        atr: float
    ) -> Tuple[float, float, float]:
        """
        Calculate TP/SL levels based on ATR.
        
        Args:
            symbol: Trading pair
            entry_price: Entry price
            side: 'BUY' or 'SELL'
            atr: ATR value
        
        Returns:
            (sl_price, tp1_price, tp2_price)
        """
        precision = self.get_symbol_precision(symbol)
        
        # ATR multipliers (from Trading Profile config)
        atr_mult_sl = 1.0   # 1R for SL
        atr_mult_tp1 = 1.5  # 1.5R for TP1
        atr_mult_tp2 = 2.5  # 2.5R for TP2
        
        if side == 'BUY':  # LONG position
            sl_price = round(entry_price - (atr * atr_mult_sl), precision)
            tp1_price = round(entry_price + (atr * atr_mult_tp1), precision)
            tp2_price = round(entry_price + (atr * atr_mult_tp2), precision)
        else:  # SHORT position
            sl_price = round(entry_price + (atr * atr_mult_sl), precision)
            tp1_price = round(entry_price - (atr * atr_mult_tp1), precision)
            tp2_price = round(entry_price - (atr * atr_mult_tp2), precision)
        
        return sl_price, tp1_price, tp2_price
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions with non-zero quantity."""
        try:
            positions = self.client.futures_position_information()
            open_positions = []
            
            for pos in positions:
                pos_amt = float(pos.get('positionAmt', 0))
                if abs(pos_amt) > 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': 'BUY' if pos_amt > 0 else 'SELL',
                        'quantity': abs(pos_amt),
                        'entry_price': entry_price,
                        'mark_price': float(pos.get('markPrice', 0)),
                        'leverage': int(pos.get('leverage', 1)),
                        'unrealized_pnl': unrealized_pnl
                    })
            
            return open_positions
        
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_open_orders(self, symbol: str) -> Dict[str, List]:
        """Get open TP/SL orders for symbol."""
        try:
            orders = self.client.futures_get_open_orders(symbol=symbol)
            
            tp_orders = []
            sl_orders = []
            
            for order in orders:
                order_type = order['type']
                if order_type in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
                    tp_orders.append(order)
                elif order_type in ['STOP_MARKET', 'STOP_LOSS', 'STOP']:
                    sl_orders.append(order)
            
            return {'tp': tp_orders, 'sl': sl_orders}
        
        except Exception as e:
            logger.error(f"Error fetching orders for {symbol}: {e}")
            return {'tp': [], 'sl': []}
    
    def validate_tpsl_direction(
        self, 
        position: Dict, 
        tp_orders: List, 
        sl_orders: List
    ) -> Tuple[bool, str]:
        """
        Validate if TP/SL orders are in correct direction.
        
        Returns:
            (is_valid, reason)
        """
        entry_price = position['entry_price']
        side = position['side']
        
        # Check TP direction
        for tp_order in tp_orders:
            tp_price = float(tp_order.get('stopPrice', 0))
            
            if side == 'BUY':  # LONG
                if tp_price <= entry_price:
                    return False, f"TP {tp_price} is BELOW entry {entry_price} (should be ABOVE for LONG)"
            else:  # SHORT
                if tp_price >= entry_price:
                    return False, f"TP {tp_price} is ABOVE entry {entry_price} (should be BELOW for SHORT)"
        
        # Check SL direction
        for sl_order in sl_orders:
            sl_price = float(sl_order.get('stopPrice', 0))
            
            if side == 'BUY':  # LONG
                if sl_price >= entry_price:
                    return False, f"SL {sl_price} is ABOVE entry {entry_price} (should be BELOW for LONG)"
            else:  # SHORT
                if sl_price <= entry_price:
                    return False, f"SL {sl_price} is BELOW entry {entry_price} (should be ABOVE for SHORT)"
        
        return True, "OK"
    
    def place_tpsl_orders(self, position: Dict, force_replace: bool = False) -> bool:
        """
        Place TP/SL orders for a position.
        
        Args:
            position: Position dict
            force_replace: Cancel existing orders first
        
        Returns:
            True if successful
        """
        symbol = position['symbol']
        entry_price = position['entry_price']
        side = position['side']
        quantity = position['quantity']
        
        try:
            # Calculate ATR
            atr = self.calculate_atr(symbol)
            if not atr:
                logger.warning(f"⚠️  Could not calculate ATR for {symbol}, using 2% estimate")
                atr = entry_price * 0.02
            
            # Calculate TP/SL levels
            sl_price, tp1_price, tp2_price = self.calculate_tpsl_levels(
                symbol, entry_price, side, atr
            )
            
            logger.info(f"📊 {symbol} ATR-based levels:")
            logger.info(f"   Entry: ${entry_price}")
            logger.info(f"   SL:    ${sl_price} (1.0R)")
            logger.info(f"   TP1:   ${tp1_price} (1.5R, 50% close)")
            logger.info(f"   TP2:   ${tp2_price} (2.5R, 30% close)")
            
            # Cancel existing orders if force_replace
            if force_replace:
                try:
                    self.client.futures_cancel_all_open_orders(symbol=symbol)
                    logger.info(f"🗑️  Cancelled all existing orders for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not cancel orders for {symbol}: {e}")
            
            # Determine order side
            if side == 'BUY':  # Close LONG with SELL
                order_side = 'SELL'
            else:  # Close SHORT with BUY
                order_side = 'BUY'
            
            # Calculate quantities for partial closes
            tp1_qty = round(quantity * 0.5, 3)  # 50%
            tp2_qty = round(quantity * 0.3, 3)  # 30%
            
            # Place SL order (full position)
            try:
                sl_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type='STOP_MARKET',
                    stopPrice=sl_price,
                    closePosition=True,  # Close entire position
                    workingType='MARK_PRICE'
                )
                logger.info(f"✅ SL placed: {symbol} @ ${sl_price} (Order ID: {sl_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Failed to place SL for {symbol}: {e}")
                return False
            
            # Place TP1 order (50%)
            try:
                tp1_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=tp1_price,
                    quantity=tp1_qty,
                    reduceOnly=True,
                    workingType='MARK_PRICE'
                )
                logger.info(f"✅ TP1 placed: {symbol} @ ${tp1_price} qty={tp1_qty} (Order ID: {tp1_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Failed to place TP1 for {symbol}: {e}")
            
            # Place TP2 order (30%)
            try:
                tp2_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=tp2_price,
                    quantity=tp2_qty,
                    reduceOnly=True,
                    workingType='MARK_PRICE'
                )
                logger.info(f"✅ TP2 placed: {symbol} @ ${tp2_price} qty={tp2_qty} (Order ID: {tp2_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Failed to place TP2 for {symbol}: {e}")
            
            logger.info(f"🛡️  {symbol} protection complete!")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error placing TP/SL for {symbol}: {e}", exc_info=True)
            return False
    
    def check_and_protect_positions(self):
        """Main check: Scan all positions and ensure TP/SL protection."""
        logger.info("=" * 80)
        logger.info("🔍 SCANNING POSITIONS FOR TP/SL PROTECTION")
        logger.info("=" * 80)
        
        # Get all open positions
        positions = self.get_open_positions()
        
        if not positions:
            logger.info("✅ No open positions")
            return
        
        logger.info(f"📊 Found {len(positions)} open position(s)")
        print()
        
        for position in positions:
            symbol = position['symbol']
            side = position['side']
            entry = position['entry_price']
            qty = position['quantity']
            pnl = position['unrealized_pnl']
            
            logger.info(f"📍 {symbol} {side} {qty} @ ${entry} (P&L: ${pnl:.2f})")
            
            # Get existing orders
            orders = self.get_open_orders(symbol)
            tp_orders = orders['tp']
            sl_orders = orders['sl']
            
            logger.info(f"   Existing: {len(tp_orders)} TP, {len(sl_orders)} SL orders")
            
            # Check if missing TP or SL
            needs_protection = False
            force_replace = False
            
            if not tp_orders and not sl_orders:
                logger.warning(f"   ⚠️  NO TP/SL ORDERS - UNPROTECTED!")
                needs_protection = True
            elif not sl_orders:
                logger.warning(f"   ⚠️  MISSING STOP LOSS!")
                needs_protection = True
            elif not tp_orders:
                logger.warning(f"   ⚠️  MISSING TAKE PROFIT!")
                needs_protection = True
            else:
                # Validate direction
                is_valid, reason = self.validate_tpsl_direction(position, tp_orders, sl_orders)
                if not is_valid:
                    logger.error(f"   ❌ WRONG TP/SL DIRECTION: {reason}")
                    needs_protection = True
                    force_replace = True
                else:
                    logger.info(f"   ✅ TP/SL direction correct")
            
            # Place orders if needed
            if needs_protection:
                logger.info(f"   🔧 Fixing {symbol}...")
                success = self.place_tpsl_orders(position, force_replace=force_replace)
                if success:
                    logger.info(f"   ✅ {symbol} protected!")
                else:
                    logger.error(f"   ❌ {symbol} protection FAILED!")
            
            print()
    
    def run(self):
        """Run continuous monitoring loop."""
        logger.info("=" * 80)
        logger.info("🛡️  POSITION PROTECTION SERVICE STARTED")
        logger.info("=" * 80)
        logger.info(f"Mode: {'TESTNET' if self.testnet else 'LIVE'}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
        print()
        
        try:
            while True:
                try:
                    self.check_and_protect_positions()
                except Exception as e:
                    logger.error(f"Error in protection loop: {e}", exc_info=True)
                
                logger.info(f"💤 Sleeping {self.check_interval}s...")
                print()
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping Position Protection Service...")
            logger.info("✅ Service stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatic Position Protection Service')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--testnet', action='store_true', help='Use testnet credentials')
    parser.add_argument('--once', action='store_true', help='Run once and exit (no loop)')
    
    args = parser.parse_args()
    
    try:
        service = PositionProtectionService(
            check_interval=args.interval,
            testnet=args.testnet
        )
        
        if args.once:
            # Run once and exit
            service.check_and_protect_positions()
            logger.info("✅ Single check complete")
        else:
            # Run continuously
            service.run()
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

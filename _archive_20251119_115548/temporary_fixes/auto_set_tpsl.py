#!/usr/bin/env python3
"""
AUTO TP/SL SETTER - HYBRID STRATEGY
Automatically sets Take Profit and Stop Loss orders for all open Binance Futures positions
Uses 50% partial exit at TP + trailing stop for remaining 50%
Runs continuously to protect new positions as they open
"""
import os
import time
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TP_PERCENT = 0.025  # 2.5% Take Profit
SL_PERCENT = 0.025  # 2.5% Stop Loss
TRAIL_PERCENT = 0.015  # 1.5% Trailing Stop
PARTIAL_EXIT = 0.5  # 50% exit at TP
CHECK_INTERVAL = 30  # Check every 30 seconds

def get_symbol_info(client, symbol):
    """Get price precision and quantity precision for a symbol from exchange info"""
    try:
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            price_precision = 2
            qty_precision = 0
            
            if price_filter:
                tick_size = float(price_filter['tickSize'])
                if tick_size >= 1:
                    price_precision = 0
                elif '.' in str(tick_size):
                    price_precision = len(str(tick_size).rstrip('0').split('.')[-1])
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                if step_size >= 1:
                    qty_precision = 0
                elif '.' in str(step_size):
                    qty_precision = len(str(step_size).rstrip('0').split('.')[-1])
            
            return price_precision, qty_precision
        return 2, 0
    except Exception as e:
        print(f"[WARNING]  Could not get symbol info for {symbol}: {e}")
        return 2, 0

def set_tpsl_for_position(client, position):
    """Set HYBRID TP/SL orders for a single position (50% partial exit + trailing stop)"""
    symbol = position['symbol']
    position_amt = float(position['positionAmt'])
    entry_price = float(position['entryPrice'])
    
    # Skip if no position
    if position_amt == 0:
        return False
    
    # Determine direction
    is_long = position_amt > 0
    
    # Get symbol info (price and quantity precision)
    price_precision, qty_precision = get_symbol_info(client, symbol)
    
    # Calculate partial exit quantity (50%)
    partial_qty = abs(position_amt) * PARTIAL_EXIT
    partial_qty = round(partial_qty, qty_precision)
    
    # Calculate TP and SL prices
    if is_long:
        tp_price = round(entry_price * (1 + TP_PERCENT), price_precision)
        sl_price = round(entry_price * (1 - SL_PERCENT), price_precision)
        trail_activation = round(entry_price * (1 + TP_PERCENT * 0.8), price_precision)  # Activate trailing at 80% of TP
        tp_side = 'SELL'
        sl_side = 'SELL'
    else:  # SHORT
        tp_price = round(entry_price * (1 - TP_PERCENT), price_precision)
        sl_price = round(entry_price * (1 + SL_PERCENT), price_precision)
        trail_activation = round(entry_price * (1 - TP_PERCENT * 0.8), price_precision)
        tp_side = 'BUY'
        sl_side = 'BUY'
    
    try:
        # Check if TP/SL orders already exist
        open_orders = client.futures_get_open_orders(symbol=symbol)
        has_tp = any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in open_orders)
        has_sl = any(o['type'] in ['STOP_MARKET', 'STOP_LOSS'] for o in open_orders)
        has_trail = any(o['type'] == 'TRAILING_STOP_MARKET' for o in open_orders)
        
        if has_tp and has_sl and has_trail:
            print(f"[OK] {symbol}: Hybrid TP/SL already set (Entry: ${entry_price}, TP: ${tp_price}, SL: ${sl_price}, Trail: {TRAIL_PERCENT*100:.1f}%)")
            return False
        
        # Cancel existing TP/SL/TRAIL orders to avoid conflicts
        for order in open_orders:
            if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP_MARKET']:
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                print(f"🗑️  Cancelled old {order['type']} order for {symbol}")
        
        # 1. Place PARTIAL TAKE_PROFIT_MARKET order (50% exit)
        if not has_tp:
            tp_order = client.futures_create_order(
                symbol=symbol,
                side=tp_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price,
                quantity=partial_qty,
                workingType='MARK_PRICE'
            )
            print(f"[OK] {symbol}: TP order (50%) placed @ ${tp_price} (+{TP_PERCENT*100:.1f}%) - Qty: {partial_qty} - Order ID: {tp_order['orderId']}")
        
        # 2. Place TRAILING_STOP_MARKET for remaining 50%
        if not has_trail:
            try:
                trail_order = client.futures_create_order(
                    symbol=symbol,
                    side=tp_side,
                    type='TRAILING_STOP_MARKET',
                    callbackRate=TRAIL_PERCENT * 100,  # Binance expects percentage as integer (1.5% = 1.5)
                    quantity=partial_qty,
                    activationPrice=trail_activation
                )
                print(f"[OK] {symbol}: Trailing stop (50%) placed @ {TRAIL_PERCENT*100:.1f}% - Activation: ${trail_activation} - Order ID: {trail_order['orderId']}")
            except Exception as trail_exc:
                # If trailing stop fails, use regular TP for remaining 50%
                print(f"[WARNING]  {symbol}: Trailing stop failed, using TP instead: {trail_exc}")
                tp_order2 = client.futures_create_order(
                    symbol=symbol,
                    side=tp_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=round(entry_price * (1 + TP_PERCENT * 1.5), price_precision),  # TP at 3.75% for rest
                    quantity=partial_qty,
                    workingType='MARK_PRICE'
                )
                print(f"[OK] {symbol}: Secondary TP (50%) placed @ ${round(entry_price * (1 + TP_PERCENT * 1.5), price_precision)} - Order ID: {tp_order2['orderId']}")
        
        # 3. Place FULL STOP_MARKET order (100% protection)
        if not has_sl:
            sl_order = client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type='STOP_MARKET',
                stopPrice=sl_price,
                closePosition=True,  # Close entire position on SL
                workingType='MARK_PRICE'
            )
            print(f"[OK] {symbol}: SL order (100%) placed @ ${sl_price} (-{SL_PERCENT*100:.1f}%) - Order ID: {sl_order['orderId']}")
        
        return True
        
    except Exception as e:
        print(f"❌ {symbol}: Failed to set hybrid TP/SL - {e}")
        return False

def main():
    """Main loop to continuously monitor and set TP/SL"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("❌ ERROR: Binance API credentials not found in .env file!")
        return
    
    print("[ROCKET] AUTO TP/SL SETTER STARTED - HYBRID STRATEGY")
    print(f"   TP: {TP_PERCENT*100}% (50% partial exit)")
    print(f"   Trailing Stop: {TRAIL_PERCENT*100}% (50% remaining)")
    print(f"   SL: {SL_PERCENT*100}% (100% protection)")
    print(f"   Check interval: {CHECK_INTERVAL}s\n")
    
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    
    while True:
        try:
            # Get all positions
            positions = client.futures_position_information()
            
            # Filter to only positions with non-zero amount
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if not open_positions:
                print(f"⏳ [{time.strftime('%H:%M:%S')}] No open positions")
            else:
                print(f"\n[CHART] [{time.strftime('%H:%M:%S')}] Found {len(open_positions)} open positions")
                for position in open_positions:
                    set_tpsl_for_position(client, position)
            
            # Wait before next check
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\n🛑 AUTO TP/SL SETTER STOPPED")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

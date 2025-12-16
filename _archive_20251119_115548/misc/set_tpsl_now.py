#!/usr/bin/env python3
"""Set TP/SL on JCTUSDT and ICPUSDT immediately with new 20x leverage settings."""
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Configuration for 20x leverage
TP_PCT = 0.04  # 4% TP
SL_PCT = 0.03  # 3% SL
TRAIL_PCT = 0.02  # 2% trailing
PARTIAL_TP = 0.5  # 50% partial exit

print("\n" + "=" * 70)
print("[SHIELD]  SETTING TP/SL ON EXISTING POSITIONS (20x leverage mode)")
print("=" * 70)

positions = client.futures_position_information()

for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    
    if amt == 0 or symbol not in ['JCTUSDT', 'ICPUSDT']:
        continue
    
    entry_price = float(p['entryPrice'])
    current_price = float(p['markPrice'])
    pnl = float(p['unRealizedProfit'])
    
    print(f"\n[CHART] {symbol}:")
    print(f"   Position: {amt}")
    print(f"   Entry: ${entry_price}")
    print(f"   Current: ${current_price}")
    print(f"   P&L: ${pnl:.2f}")
    
    # Get price precision
    try:
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize'])
        price_precision = len(str(tick_size).rstrip('0').split('.')[-1]) if '.' in str(tick_size) else 0
    except:
        price_precision = 4
    
    # Cancel existing orders
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
            print(f"   🗑️  Cancelled old order: {order['type']}")
    except Exception as e:
        print(f"   [WARNING]  Could not cancel orders: {e}")
    
    # Calculate TP/SL prices
    if amt > 0:  # LONG position
        tp_price = round(entry_price * (1 + TP_PCT), price_precision)
        sl_price = round(entry_price * (1 - SL_PCT), price_precision)
        tp_side = 'SELL'
        sl_side = 'SELL'
    else:  # SHORT position
        tp_price = round(entry_price * (1 - TP_PCT), price_precision)
        sl_price = round(entry_price * (1 + SL_PCT), price_precision)
        tp_side = 'BUY'
        sl_side = 'BUY'
    
    # Calculate partial quantity
    partial_qty = abs(amt) * PARTIAL_TP
    remaining_qty = abs(amt) - partial_qty
    
    # Round to proper precision
    qty_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    step_size = float(qty_filter['stepSize']) if qty_filter else 0.01
    
    def round_qty(qty, step):
        """Round quantity to step size."""
        if step >= 1:
            return int(qty)
        precision = len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0
        return round(qty, precision)
    
    partial_qty = round_qty(partial_qty, step_size)
    remaining_qty = round_qty(remaining_qty, step_size)
    
    print(f"\n   [TARGET] HYBRID STRATEGY (20x leverage):")
    print(f"   • TP Price: ${tp_price} (+{TP_PCT*100:.1f}%)")
    print(f"   • SL Price: ${sl_price} (-{SL_PCT*100:.1f}%)")
    print(f"   • Partial Exit: {partial_qty} @ TP")
    print(f"   • Trailing: {remaining_qty} @ {TRAIL_PCT*100:.1f}%")
    
    try:
        # 1. Partial TP order (50% at TP price)
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=tp_price,
            quantity=partial_qty,
            workingType='MARK_PRICE',
            reduceOnly=True
        )
        print(f"   [OK] TP order: {partial_qty} @ ${tp_price}")
        
        # 2. Trailing stop for remaining 50%
        trail_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TRAILING_STOP_MARKET',
            quantity=remaining_qty,
            callbackRate=TRAIL_PCT * 100,
            workingType='MARK_PRICE',
            reduceOnly=True
        )
        print(f"   [OK] Trailing: {remaining_qty} @ {TRAIL_PCT*100:.1f}%")
        
        # 3. Stop loss for full position (backup protection)
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type='STOP_MARKET',
            stopPrice=sl_price,
            closePosition=True,
            workingType='MARK_PRICE'
        )
        print(f"   [OK] SL order: Full position @ ${sl_price}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("[OK] TP/SL SET ON EXISTING POSITIONS!")
print("=" * 70 + "\n")

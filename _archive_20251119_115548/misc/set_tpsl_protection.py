#!/usr/bin/env python3
"""Set TP/SL orders on all open Binance Futures positions."""

from binance.client import Client
import os

client = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])

# TP/SL settings from docker-compose
TP_PCT = 0.005  # 0.5% take profit
SL_PCT = 0.0075  # 0.75% stop loss

print("\n[SHIELD]  SETTING TP/SL PROTECTION ON ALL POSITIONS\n")

# Get all open positions
positions = client.futures_position_information()
open_pos = [p for p in positions if float(p['positionAmt']) != 0]

print(f"Found {len(open_pos)} open positions\n")

for p in open_pos:
    symbol = p['symbol']
    size = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    side = 'LONG' if size > 0 else 'SHORT'
    
    print(f"{symbol:12} | {side:5} | {abs(size):8.2f} @ ${entry:8.4f}")
    
    # Calculate TP and SL prices
    if size > 0:  # LONG position
        tp_price = entry * (1 + TP_PCT)
        sl_price = entry * (1 - SL_PCT)
        tp_side = 'SELL'
        sl_side = 'SELL'
    else:  # SHORT position
        tp_price = entry * (1 - TP_PCT)
        sl_price = entry * (1 + SL_PCT)
        tp_side = 'BUY'
        sl_side = 'BUY'
    
    print(f"  TP: ${tp_price:.4f} (+{TP_PCT*100:.1f}%)")
    print(f"  SL: ${sl_price:.4f} (-{SL_PCT*100:.2f}%)")
    
    try:
        # Cancel any existing orders for this symbol
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
            print(f"  🗑️  Cancelled existing order {order['orderId']}")
        
        # Place TAKE_PROFIT_MARKET order
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TAKE_PROFIT_MARKET',
            quantity=abs(size),
            stopPrice=round(tp_price, 2),
            reduceOnly=True,
            workingType='MARK_PRICE'
        )
        print(f"  [OK] TP order placed: {tp_order['orderId']}")
        
        # Place STOP_MARKET order
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type='STOP_MARKET',
            quantity=abs(size),
            stopPrice=round(sl_price, 2),
            reduceOnly=True,
            workingType='MARK_PRICE'
        )
        print(f"  [OK] SL order placed: {sl_order['orderId']}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()

print("[OK] All TP/SL orders placed!\n")

# Verify
print("[CHART] Verifying open orders:\n")
for p in open_pos:
    symbol = p['symbol']
    orders = client.futures_get_open_orders(symbol=symbol)
    print(f"{symbol:12} | {len(orders)} orders")
    for order in orders:
        print(f"  • {order['type']:20} | {order['side']:4} | ${float(order['stopPrice']):.4f}")

print()

#!/usr/bin/env python3
import os
from binance.client import Client

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
pos = client.futures_position_information(symbol='XANUSDT')[0]
entry = float(pos['entryPrice'])
mark = float(pos['markPrice'])
size = float(pos['positionAmt'])

print(f"XANUSDT LONG:")
print(f"  Entry: ${entry}")
print(f"  Current: ${mark}")
print(f"  Size: {size}")

# Get exchange info
exchange_info = client.futures_exchange_info()
symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == 'XANUSDT'), None)
filters = {f['filterType']: f for f in symbol_info['filters']}
tick_size = float(filters['PRICE_FILTER']['tickSize'])
precision = len(str(tick_size).split('.')[-1])

print(f"  Precision: {precision}")

# Calculate SL from CURRENT price (not entry)
sl_pct = 0.0075
sl_from_current = round(mark * (1 - sl_pct), precision)
sl_from_entry = round(entry * (1 - sl_pct), precision)

print(f"  SL from entry: ${sl_from_entry}")
print(f"  SL from current: ${sl_from_current}")

# Use the one that's below current price
sl_price = min(sl_from_current, sl_from_entry)
print(f"  Using SL: ${sl_price}")

try:
    sl_order = client.futures_create_order(
        symbol='XANUSDT',
        side='SELL',
        type='STOP_MARKET',
        stopPrice=sl_price,
        closePosition=True,
        workingType='MARK_PRICE'
    )
    print(f"[OK] SL order placed: {sl_order['orderId']}")
except Exception as e:
    print(f"❌ Failed: {e}")

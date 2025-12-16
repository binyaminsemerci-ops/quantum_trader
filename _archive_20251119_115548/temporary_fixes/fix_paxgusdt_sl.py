#!/usr/bin/env python3
"""
Fix missing SL for PAXGUSDT position
"""
import os
from binance.client import Client

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

symbol = "PAXGUSDT"

# Get position
positions = client.futures_position_information(symbol=symbol)
pos = positions[0]
size = float(pos['positionAmt'])
entry = float(pos['entryPrice'])

print(f"{symbol} {'LONG' if size > 0 else 'SHORT'}: Size={size}, Entry=${entry}")

# Get precision
exchange_info = client.futures_exchange_info()
symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
filters = {f['filterType']: f for f in symbol_info['filters']}
tick_size = float(filters['PRICE_FILTER']['tickSize'])
if tick_size >= 1:
    precision = 0
elif '.' in str(tick_size):
    precision = len(str(tick_size).rstrip('0').split('.')[-1])
else:
    precision = 0
print(f"tickSize: {tick_size}, precision: {precision} decimals")

# Calculate SL (0.75% from entry)
sl_pct = 0.0075
if size > 0:  # LONG
    sl_price = round(entry * (1 - sl_pct), precision)
    sl_side = 'SELL'
else:  # SHORT
    sl_price = round(entry * (1 + sl_pct), precision)
    sl_side = 'BUY'

print(f"SL Price: {sl_price}")

# Place SL order
try:
    sl_order = client.futures_create_order(
        symbol=symbol,
        side=sl_side,
        type='STOP_MARKET',
        stopPrice=sl_price,
        closePosition=True,
        workingType='MARK_PRICE'
    )
    print(f"[OK] SL order placed: {sl_order['orderId']}")
except Exception as e:
    print(f"❌ Failed: {e}")

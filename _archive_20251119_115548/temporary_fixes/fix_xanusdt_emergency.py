#!/usr/bin/env python3
import os
from binance.client import Client

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

symbol = "XANUSDT"

# Get actual position from Binance
pos = client.futures_position_information(symbol=symbol)[0]
size = float(pos['positionAmt'])
entry = float(pos['entryPrice'])
mark = float(pos['markPrice'])

print(f"{symbol} LONG:")
print(f"  Size: {size}")
print(f"  Entry: ${entry:.8f}")
print(f"  Current: ${mark:.8f}")

# Get precision
exchange_info = client.futures_exchange_info()
symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
filters = {f['filterType']: f for f in symbol_info['filters']}
tick_size = float(filters['PRICE_FILTER']['tickSize'])
precision = len(str(tick_size).split('.')[-1])

print(f"  Precision: {precision}")

# Calculate TP/SL from actual entry
tp_pct = 0.076  # 7.6% from AI
sl_pct = 0.03   # 3% from AI

tp_price = round(entry * (1 + tp_pct), precision)
sl_price = round(entry * (1 - sl_pct), precision)

print(f"\n[CHART] TP/SL from actual entry:")
print(f"  TP: ${tp_price:.8f} (+{tp_pct*100:.1f}%)")
print(f"  SL: ${sl_price:.8f} (-{sl_pct*100:.1f}%)")

# Check if SL would trigger immediately
if mark < sl_price:
    print(f"\n[WARNING]  WARNING: Current price ${mark:.8f} is BELOW SL ${sl_price:.8f}!")
    print(f"   Using current price as basis for SL instead...")
    sl_price = round(mark * (1 - 0.01), precision)  # 1% below current
    print(f"   New SL: ${sl_price:.8f}")

# Place TP order
try:
    tp_order = client.futures_create_order(
        symbol=symbol,
        side='SELL',
        type='TAKE_PROFIT_MARKET',
        stopPrice=tp_price,
        closePosition=True,
        workingType='MARK_PRICE'
    )
    print(f"\n[OK] TP order placed: {tp_order['orderId']} @ ${tp_price:.8f}")
except Exception as e:
    print(f"\n❌ TP failed: {e}")

# Place SL order
try:
    sl_order = client.futures_create_order(
        symbol=symbol,
        side='SELL',
        type='STOP_MARKET',
        stopPrice=sl_price,
        closePosition=True,
        workingType='MARK_PRICE'
    )
    print(f"[OK] SL order placed: {sl_order['orderId']} @ ${sl_price:.8f}")
except Exception as e:
    print(f"❌ SL failed: {e}")

# Update trade state with correct entry
import json
trade_state_path = '/app/backend/data/trade_state.json'
with open(trade_state_path, 'r') as f:
    trade_state = json.load(f)

if symbol in trade_state:
    trade_state[symbol]['avg_entry'] = entry
    trade_state[symbol]['peak'] = entry
    
    with open(trade_state_path, 'w') as f:
        json.dump(trade_state, f, indent=2)
    
    print(f"\n[OK] Trade state updated with correct entry: ${entry:.8f}")

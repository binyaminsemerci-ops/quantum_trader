from binance.client import Client
import os

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Get PUMPUSDT position
positions = client.futures_position_information(symbol='PUMPUSDT')
pos = next((p for p in positions if float(p['positionAmt']) != 0), None)

if not pos:
    print("❌ No PUMPUSDT position found")
    exit()

entry = float(pos['entryPrice'])
qty = float(pos['positionAmt'])
side = 'LONG' if qty > 0 else 'SHORT'

print(f"PUMPUSDT {side}: Size={qty}, Entry=${entry}")

# Get precision
info = client.futures_exchange_info()
symbol_info = next(s for s in info['symbols'] if s['symbol'] == 'PUMPUSDT')
price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
tick_size = float(price_filter['tickSize'])

if tick_size >= 1:
    precision = 0
else:
    precision = len(str(tick_size).rstrip('0').split('.')[-1])

print(f"tickSize: {tick_size}, precision: {precision} decimals")

# Calculate SL (0.75% below entry for LONG)
sl_pct = 0.0075
if side == 'LONG':
    sl_price = round(entry * (1 - sl_pct), precision)
    order_side = 'SELL'
else:
    sl_price = round(entry * (1 + sl_pct), precision)
    order_side = 'BUY'

print(f"SL Price: {sl_price}")

# Place SL order
try:
    sl_order = client.futures_create_order(
        symbol='PUMPUSDT',
        side=order_side,
        type='STOP_MARKET',
        stopPrice=sl_price,
        closePosition=True,
        workingType='MARK_PRICE'
    )
    print(f"[OK] SL order placed: {sl_order['orderId']}")
except Exception as e:
    print(f"❌ SL order failed: {e}")

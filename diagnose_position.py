#!/usr/bin/env python3
"""Quick diagnosis of current position and recent orders"""
import os
import sys
sys.path.insert(0, '/home/qt/quantum_trader')

# Set credentials
os.environ['BINANCE_API_KEY'] = os.getenv('BINANCE_API_KEY', '')
os.environ['BINANCE_API_SECRET'] = os.getenv('BINANCE_API_SECRET', '')

from binance.client import Client

client = Client(
    os.environ['BINANCE_API_KEY'],
    os.environ['BINANCE_API_SECRET'],
    testnet=True
)

print("=" * 60)
print("CURRENT POSITION STATUS")
print("=" * 60)

# Get position
positions = client.futures_position_information(symbol="METISUSDT")
for pos in positions:
    amt = float(pos['positionAmt'])
    if abs(amt) > 0:
        print(f"Position Side: {pos['positionSide']}")
        print(f"Amount: {amt} METIS")
        print(f"Entry Price: {pos['entryPrice']}")
        print(f"Mark Price: {pos['markPrice']}")
        print(f"Unrealized PnL: {pos['unRealizedProfit']} USDT")
        print(f"Leverage: {pos['leverage']}x")
        print()

print("=" * 60)
print("RECENT ORDERS (last 20)")
print("=" * 60)

# Get recent orders
orders = client.futures_get_all_orders(symbol="METISUSDT", limit=20)
orders.reverse()  # Most recent first

filled_count = 0
for order in orders:
    if order['status'] == 'FILLED':
        filled_count += 1
        print(f"{order['updateTime']} | {order['side']} {order['positionSide']} | "
              f"Qty: {order['executedQty']} @ {order['avgPrice']} | "
              f"OrderID: {order['orderId']}")

print()
print(f"Total FILLED orders: {filled_count}")

print("=" * 60)
print("OPEN ORDERS (SL/TP)")
print("=" * 60)

open_orders = client.futures_get_open_orders(symbol="METISUSDT")
if open_orders:
    for order in open_orders:
        print(f"{order['type']} | Side: {order['side']} {order['positionSide']} | "
              f"Stop Price: {order.get('stopPrice', 'N/A')} | "
              f"OrderID: {order['orderId']}")
else:
    print("⚠️ NO OPEN SL/TP ORDERS!")

print("=" * 60)

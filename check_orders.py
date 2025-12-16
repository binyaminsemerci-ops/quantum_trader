#!/usr/bin/env python
"""Check all open TP/SL orders"""
from binance.client import Client
import os

client = Client(
    os.getenv('BINANCE_TEST_API_KEY'), 
    os.getenv('BINANCE_TEST_API_SECRET'), 
    testnet=True
)

orders = client.futures_get_open_orders()
print(f'\n📋 Total open orders: {len(orders)}\n')

if not orders:
    print("❌ NO TP/SL ORDERS FOUND - All positions are UNPROTECTED!\n")
else:
    for order in orders:
        symbol = order['symbol']
        order_type = order['type']
        side = order['side']
        price = order.get('stopPrice') or order.get('price', 'N/A')
        print(f"{symbol}: {order_type} {side} @ {price}")

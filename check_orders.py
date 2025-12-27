#!/usr/bin/env python
"""Check all open TP/SL orders"""
from binance.client import Client
import os

# Use environment variables from Position Monitor container
api_key = os.getenv('BINANCE_API_KEY', 'IsY3mFpko7Z8joZr8clWwpJZuZcFdAtnDBy4g4ULQu827Gf6kJPAPS9VyVOVrR0r')
api_secret = os.getenv('BINANCE_API_SECRET', 'tEKYWf77tqSOArfgeSqgVwiO62bjro8D1VMaEvXBwcUOQuRxmCdxrtTvAXy7ZKSE')

client = Client(api_key, api_secret, testnet=True)

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

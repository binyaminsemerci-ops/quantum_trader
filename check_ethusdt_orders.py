#!/usr/bin/env python3
"""Check ETHUSDT open orders on Binance"""
import os
from binance.client import Client

# Initialize Binance client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

client = Client(api_key, api_secret, testnet=True)

# Get open orders
orders = client.futures_get_open_orders(symbol='ETHUSDT')

print(f"\n📊 ETHUSDT Open Orders: {len(orders)} total\n")

if orders:
    for o in orders:
        order_type = o['type']
        side = o['side']
        price = o.get('stopPrice') or o.get('price') or 'N/A'
        qty = o.get('origQty', 'N/A')
        position_side = o.get('positionSide', 'BOTH')
        
        print(f"  {order_type}: {side} @ ${price}")
        print(f"    Qty: {qty}, PositionSide: {position_side}")
        print(f"    OrderId: {o['orderId']}")
        print()
else:
    print("  ❌ No open orders found for ETHUSDT!")

#!/usr/bin/env python3
"""
Quick test: Can we create LIMIT orders successfully?
"""

import os
from binance.client import Client

# Initialize client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

symbol = 'SOLUSDT'

# Get current price
ticker = client.futures_symbol_ticker(symbol=symbol)
current_price = float(ticker['price'])

# Try to create a LIMIT order far from current price (won't execute)
test_price = round(current_price * 0.5, 2)  # 50% below current price

print(f"Current {symbol} price: ${current_price:.2f}")
print(f"Attempting LIMIT BUY at ${test_price:.2f} (won't execute)...")

try:
    order = client.futures_create_order(
        symbol=symbol,
        side='BUY',
        type='LIMIT',
        timeInForce='GTC',
        quantity=1.0,
        price=test_price
    )
    
    order_id = order.get('orderId')
    print(f"✅ SUCCESS: Created LIMIT order {order_id}")
    
    # Clean up - cancel the order
    client.futures_cancel_order(symbol=symbol, orderId=order_id)
    print(f"✅ Cancelled order {order_id}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

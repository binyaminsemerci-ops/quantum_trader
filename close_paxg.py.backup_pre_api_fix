#!/usr/bin/env python3
"""Close PAXGUSDT position to free up a slot"""
import os
from binance.client import Client

# Testnet API credentials
api_key = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")

client = Client(api_key, api_secret, testnet=True)

symbol = "PAXGUSDT"

# Get current position
positions = client.futures_position_information(symbol=symbol)
pos = positions[0]
amt = float(pos['positionAmt'])

if amt == 0:
    print(f"[OK] No open position for {symbol}")
else:
    side = "SELL" if amt > 0 else "BUY"  # Opposite side to close
    qty = abs(amt)
    
    print(f"📤 Closing {symbol}: {side} {qty}")
    
    order = client.futures_create_order(
        symbol=symbol,
        side=side,
        type='MARKET',
        quantity=qty,
        reduceOnly=True
    )
    
    print(f"[OK] Order placed: {order['orderId']}")
    print(f"   Status: {order['status']}")
    print(f"   Side: {order['side']}")
    print(f"   Qty: {order['origQty']}")

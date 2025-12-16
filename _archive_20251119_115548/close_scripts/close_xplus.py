#!/usr/bin/env python3
"""Close XPLUSDT position."""
from binance.client import Client
import os

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
positions = [p for p in client.futures_position_information(symbol='XPLUSDT') if float(p['positionAmt']) != 0]

if positions:
    p = positions[0]
    size = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    pnl = float(p['unRealizedProfit'])
    
    print(f"\nXPLUSDT Position:")
    print(f"  Size: {size}")
    print(f"  Entry: ${entry}")
    print(f"  PNL: ${pnl}")
    print(f"\n🔥 Closing XPLUSDT at market...")
    
    # Close position
    side = 'SELL' if size > 0 else 'BUY'
    result = client.futures_create_order(
        symbol='XPLUSDT',
        side=side,
        type='MARKET',
        quantity=abs(size),
        reduceOnly=True
    )
    
    print(f"[OK] Closed! Order ID: {result['orderId']}")
    print(f"[MONEY] Loss accepted: ${pnl:.2f}")
else:
    print("No XPLUSDT position found")

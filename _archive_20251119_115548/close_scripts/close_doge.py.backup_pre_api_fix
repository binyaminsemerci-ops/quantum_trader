#!/usr/bin/env python3
"""Close DOGEUSDT position to lock in profit."""

from binance.client import Client
import os

client = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])

# Get DOGE position
positions = client.futures_position_information(symbol='DOGEUSDT')
pos = positions[0]
size = float(pos['positionAmt'])
entry = float(pos['entryPrice'])
pnl = float(pos['unRealizedProfit'])

print(f'\nDOGEUSDT Position:')
print(f'  Size: {size}')
print(f'  Entry: ${entry:.4f}')
print(f'  PNL: ${pnl:+.2f}')

if size != 0:
    print(f'\n🔥 Closing DOGEUSDT at market...')
    result = client.futures_create_order(
        symbol='DOGEUSDT',
        side='SELL',
        type='MARKET',
        quantity=abs(size),
        reduceOnly=True
    )
    print(f'[OK] Closed! Order ID: {result["orderId"]}')
    print(f'[MONEY] Profit locked: ${pnl:+.2f}')
else:
    print('[WARNING]  No position to close')

print()

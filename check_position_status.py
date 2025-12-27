#!/usr/bin/env python3
from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)
positions = [p for p in c.futures_position_information() if float(p['positionAmt']) != 0]
orders = c.futures_get_open_orders()

print(f'Open positions: {len(positions)}')
print(f'Total open orders: {len(orders)}')

# Group orders by symbol
by_symbol = {}
for order in orders:
    symbol = order['symbol']
    by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

print('\nOrders by symbol:')
for symbol in sorted(by_symbol.keys()):
    count = by_symbol[symbol]
    status = '❌ LIMIT!' if count >= 10 else '⚠️' if count >= 8 else '✅'
    print(f'  {status} {symbol}: {count} orders')

print(f'\nPositions: {[p["symbol"] for p in positions]}')

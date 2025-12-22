#!/usr/bin/env python3
"""Check and clean up stop orders on Binance"""

from binance.client import Client
import os

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

# Get all open orders
orders = client.futures_get_open_orders()
print(f'📊 Total open orders: {len(orders)}')

# Group by type
by_type = {}
for order in orders:
    order_type = order.get('type', 'UNKNOWN')
    by_type[order_type] = by_type.get(order_type, 0) + 1

print('\n📋 Orders by type:')
for t, count in sorted(by_type.items()):
    print(f'  {t}: {count}')

# Show stop orders
stop_orders = [o for o in orders if 'STOP' in o.get('type', '')]
print(f'\n🛑 Stop orders: {len(stop_orders)}')

# Group by symbol
by_symbol = {}
for order in stop_orders:
    symbol = order['symbol']
    by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

print('\n📈 Stop orders by symbol:')
for symbol, count in sorted(by_symbol.items(), key=lambda x: x[1], reverse=True):
    print(f'  {symbol}: {count} stop orders')

# Check Binance limit (usually 10 stop orders per symbol)
print('\n⚠️  Symbols with many stop orders:')
for symbol, count in by_symbol.items():
    if count >= 8:
        print(f'  ❌ {symbol}: {count}/10 (NEAR LIMIT!)')
    elif count >= 5:
        print(f'  ⚠️  {symbol}: {count}/10')

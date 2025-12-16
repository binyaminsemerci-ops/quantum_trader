#!/usr/bin/env python
"""Check and clean all open orders"""
from binance.client import Client
import os

client = Client(
    os.getenv('BINANCE_TEST_API_KEY'),
    os.getenv('BINANCE_TEST_API_SECRET'),
    testnet=True
)

# Get all open orders
orders = client.futures_get_open_orders()
print(f"\n📋 Total open orders: {len(orders)}\n")

# Group by symbol
by_symbol = {}
for order in orders:
    symbol = order['symbol']
    if symbol not in by_symbol:
        by_symbol[symbol] = []
    by_symbol[symbol].append(order)

print(f"Orders by symbol:")
for symbol, sym_orders in sorted(by_symbol.items()):
    print(f"\n{symbol}: {len(sym_orders)} orders")
    for o in sym_orders:
        print(f"  - {o['type']} {o['side']} @ {o.get('stopPrice') or o.get('price', 'N/A')}")

# Clean orphaned orders (symbols with no position)
positions = client.futures_position_information()
open_symbols = {p['symbol'] for p in positions if float(p['positionAmt']) != 0}

orphaned = [s for s in by_symbol.keys() if s not in open_symbols]

if orphaned:
    print(f"\n🗑️  Found {len(orphaned)} symbols with orphaned orders:")
    for symbol in orphaned:
        print(f"  - {symbol}: {len(by_symbol[symbol])} orders (NO POSITION)")
        
    response = input("\nDelete all orphaned orders? (yes/no): ")
    if response.lower() == 'yes':
        deleted = 0
        for symbol in orphaned:
            try:
                result = client.futures_cancel_all_open_orders(symbol=symbol)
                deleted += len(by_symbol[symbol])
                print(f"  ✅ Deleted {len(by_symbol[symbol])} orders for {symbol}")
            except Exception as e:
                print(f"  ❌ Failed to delete {symbol}: {e}")
        print(f"\n✅ Total deleted: {deleted} orders")
else:
    print("\n✅ No orphaned orders found")

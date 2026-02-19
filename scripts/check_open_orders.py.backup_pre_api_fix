#!/usr/bin/env python3
"""Check open orders on Binance Testnet"""

from binance.client import Client
import os

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print("\n" + "="*70)
print("📋 ÅPNE ORDRE PÅ BINANCE TESTNET")
print("="*70 + "\n")

# Get open orders
open_orders = client.futures_get_open_orders(symbol='BTCUSDT')

print(f"Antall åpne ordre: {len(open_orders)}\n")

if open_orders:
    for i, order in enumerate(open_orders, 1):
        print(f"Ordre #{i}:")
        print(f"  Order ID: {order['orderId']}")
        print(f"  Type: {order['type']}")
        print(f"  Side: {order['side']}")
        print(f"  Mengde: {order['origQty']} BTC")
        
        if 'stopPrice' in order and order['stopPrice'] != '0':
            print(f"  Stop Price: ${float(order['stopPrice']):,.2f}")
        elif 'price' in order and order['price'] != '0':
            print(f"  Limit Price: ${float(order['price']):,.2f}")
        
        print(f"  Status: {order['status']}")
        print()
else:
    print("✅ Ingen åpne ordre\n")

# Get current position
positions = client.futures_position_information(symbol='BTCUSDT')

print("="*70)
print("📊 CURRENT POSITION")
print("="*70 + "\n")

for pos in positions:
    if float(pos['positionAmt']) != 0:
        print(f"Symbol: {pos['symbol']}")
        print(f"Posisjon: {pos['positionAmt']} BTC")
        print(f"Entry Price: ${float(pos['entryPrice']):,.2f}")
        print(f"Unrealized P&L: ${float(pos['unRealizedProfit']):.2f}")
        print()

print("="*70 + "\n")

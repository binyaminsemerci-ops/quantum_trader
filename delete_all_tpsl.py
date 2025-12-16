#!/usr/bin/env python3
"""Cancel all TP/SL orders to trigger recalculation."""
import os
import sys
sys.path.insert(0, '/app')

from binance.client import Client

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print(f"API Key: {api_key[:10] if api_key else 'MISSING'}...")
    print(f"API Secret: {api_secret[:10] if api_secret else 'MISSING'}...")
    sys.exit(1)

client = Client(api_key, api_secret)

print("\n🗑️  SLETTER ALLE TP/SL ORDERS\n")

# Get all open positions
positions = client.futures_position_information()
open_pos = [p for p in positions if float(p['positionAmt']) != 0]

print(f"Fant {len(open_pos)} åpne posisjoner\n")

for pos in open_pos:
    symbol = pos['symbol']
    print(f"[CHART] {symbol}:")
    
    orders = client.futures_get_open_orders(symbol=symbol)
    cancelled = 0
    
    for order in orders:
        if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'TRAILING_STOP_MARKET']:
            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
            print(f"   [OK] Slettet {order['type']} order {order['orderId']}")
            cancelled += 1
    
    print(f"   Total: {cancelled} orders slettet\n")

print(f"[OK] Position monitor vil oppdage ubes kyttede posisjoner og sette nye TP/SL!")
print(f"Vent 30 sekunder...")

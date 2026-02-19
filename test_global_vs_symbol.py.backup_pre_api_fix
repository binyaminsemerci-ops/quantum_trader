#!/usr/bin/env python3
"""
Test if it's a GLOBAL limit or PER-SYMBOL limit
"""

from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print("=" * 80)
print("Testing: GLOBAL vs PER-SYMBOL limit")
print("=" * 80)

# Test on a FRESH symbol that we don't have position in
all_symbols = c.futures_exchange_info()['symbols']
test_candidates = ['LTCUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT']

print("\nTrying to create orders on symbols WITHOUT positions:")

for test_sym in test_candidates:
    symbol_info = next((s for s in all_symbols if s['symbol'] == test_sym), None)
    if not symbol_info:
        continue
    
    try:
        # Get current price
        ticker = c.futures_symbol_ticker(symbol=test_sym)
        price = float(ticker['price'])
        
        # Try to create a small test order
        test_price = round(price * 1.10, 2)
        test_qty = 1.0
        
        print(f"\n{test_sym}:")
        print(f"  Attempting SELL TAKE_PROFIT @ ${test_price}")
        
        order = c.futures_create_order(
            symbol=test_sym,
            side='SELL',
            type='TAKE_PROFIT_MARKET',
            stopPrice=test_price,
            quantity=test_qty,
            workingType='MARK_PRICE',
            positionSide='BOTH',
            reduceOnly=False  # Allow opening position
        )
        
        print(f"  ✅ SUCCESS! Order ID: {order['orderId']}")
        print(f"  This proves it's PER-SYMBOL limit, not global!")
        
        # Cancel it
        c.futures_cancel_order(symbol=test_sym, orderId=order['orderId'])
        print(f"  Cancelled test order")
        break
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("If we CAN create orders on fresh symbols:")
print("  → This is a PER-SYMBOL issue with stuck orders")
print("  → Those specific symbols have invisible orders blocking new ones")
print("  → FIX: Close positions on those symbols, wait 24h, or use production API")
print()
print("If we CANNOT create orders on ANY symbol:")
print("  → This is a GLOBAL ACCOUNT limit")
print("  → FIX: Contact Binance support or wait for testnet reset")
print("=" * 80)

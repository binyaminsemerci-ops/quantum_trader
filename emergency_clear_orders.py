#!/usr/bin/env python3
"""
EMERGENCY FIX: Cancel ALL orders including hidden/stuck ones
"""

from binance.client import Client
import os
import time

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

problem_symbols = ['SOLUSDT', 'DOTUSDT', 'BNBUSDT', 'ADAUSDT', 'ETHUSDT', 'XRPUSDT', 'BTCUSDT']

print("=" * 80)
print("EMERGENCY: Cancelling ALL orders (including hidden ones)")
print("=" * 80)

for symbol in problem_symbols:
    print(f"\n{symbol}:")
    try:
        # Use cancelAllOpenOrders - this cancels EVERYTHING including conditional/hidden
        result = c.futures_cancel_all_open_orders(symbol=symbol)
        print(f"  ✓ Cancelled all orders: {result}")
        time.sleep(0.5)
    except Exception as e:
        error_str = str(e)
        if 'No such open order' in error_str or 'Unknown order sent' in error_str:
            print(f"  ✓ No orders to cancel (expected)")
        else:
            print(f"  ✗ Error: {e}")

print("\n" + "=" * 80)
print("Waiting 5 seconds for Binance to process...")
print("=" * 80)
time.sleep(5)

print("\nTesting if we can now create orders...")

# Test on first problem symbol
test_symbol = 'SOLUSDT'
try:
    pos = [p for p in c.futures_position_information(symbol=test_symbol) if float(p['positionAmt']) != 0][0]
    amt = float(pos['positionAmt'])
    side = 'BUY' if amt < 0 else 'SELL'
    test_price = float(pos['markPrice']) * (0.90 if amt < 0 else 1.10)
    
    # Round properly
    test_price = round(test_price, 2)  # SOLUSDT uses 0.01 tick
    test_qty = round(abs(amt) * 0.05)  # 5% test, SOLUSDT uses 1.0 step
    
    if test_qty < 1:
        test_qty = 1.0
    
    print(f"\nCreating test order on {test_symbol}:")
    print(f"  {side} {test_qty} @ ${test_price}")
    
    order = c.futures_create_order(
        symbol=test_symbol,
        side=side,
        type='TAKE_PROFIT_MARKET',
        stopPrice=test_price,
        quantity=test_qty,
        workingType='MARK_PRICE',
        positionSide='BOTH'
    )
    
    print(f"  ✅ SUCCESS! Order created: {order['orderId']}")
    
    # Cancel it
    c.futures_cancel_order(symbol=test_symbol, orderId=order['orderId'])
    print(f"  ✅ Test order cancelled")
    
    print("\n" + "=" * 80)
    print("✅ FIX SUCCESSFUL! Orders can now be placed.")
    print("=" * 80)
    
except Exception as e:
    print(f"  ✗ Still failing: {e}")
    print("\n" + "=" * 80)
    print("⚠️  Issue persists. Recommendations:")
    print("   1. Wait 60 seconds and try again")
    print("   2. Contact Binance support about stuck orders")
    print("   3. Switch to production API")
    print("=" * 80)

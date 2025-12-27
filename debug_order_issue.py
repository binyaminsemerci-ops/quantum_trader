#!/usr/bin/env python3
"""Debug order limit issue"""

from binance.client import Client
import os
import time

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print("=" * 60)
print("CHECKING ALL ORDERS (including hidden/conditional)")
print("=" * 60)

# Get all open orders
all_orders = c.futures_get_open_orders()
print(f"\n📊 Regular open orders: {len(all_orders)}")

# Get positions
positions = [p for p in c.futures_position_information() if float(p['positionAmt']) != 0]
print(f"📊 Open positions: {len(positions)}")

# For each position, check detailed info
print("\n" + "=" * 60)
print("PER-SYMBOL DETAILED CHECK")
print("=" * 60)

for pos in positions:
    symbol = pos['symbol']
    amt = float(pos['positionAmt'])
    
    # Get ALL orders for this symbol
    symbol_orders = c.futures_get_open_orders(symbol=symbol)
    
    print(f"\n🔍 {symbol}: position={amt:.4f}")
    print(f"   Open orders: {len(symbol_orders)}")
    
    if symbol_orders:
        for order in symbol_orders:
            print(f"   - {order['type']}: {order['origQty']} @ {order.get('stopPrice', order.get('price', 'N/A'))}")
    
    # Try to create a test order to see what happens
    try:
        side = 'SELL' if amt > 0 else 'BUY'
        test_price = float(pos['markPrice']) * (1.05 if amt > 0 else 0.95)
        
        print(f"   Testing order creation: {side} TAKE_PROFIT_MARKET @ {test_price:.8f}")
        
        # Create test TP order
        test_order = c.futures_create_order(
            symbol=symbol,
            side=side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=test_price,
            quantity=abs(amt) * 0.1,  # Only 10% for test
            workingType='MARK_PRICE',
            positionSide='BOTH'
        )
        
        print(f"   ✅ Test order created: {test_order['orderId']}")
        
        # Immediately cancel it
        c.futures_cancel_order(symbol=symbol, orderId=test_order['orderId'])
        print(f"   ✅ Test order cancelled")
        
        time.sleep(0.2)  # Small delay
        
    except Exception as e:
        print(f"   ❌ Test order FAILED: {e}")
        if '-4045' in str(e):
            print(f"   ⚠️  FOUND IT! This symbol hits the limit!")
            # Try to get more info
            print(f"   Checking for hidden orders...")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total positions: {len(positions)}")
print(f"Total visible orders: {len(all_orders)}")
print("\nIf you see -4045 errors above, that symbol has hidden/stuck orders")

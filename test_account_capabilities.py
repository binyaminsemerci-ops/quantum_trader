#!/usr/bin/env python3
"""
Check if we can do ANYTHING on this account
"""

from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print("=" * 80)
print("ACCOUNT CAPABILITIES TEST")
print("=" * 80)

# 1. Can we query?
print("\n[TEST 1] Can we query account info?")
try:
    account = c.futures_account()
    print(f"✅ YES - Balance: ${float(account['totalWalletBalance']):.2f}")
except Exception as e:
    print(f"❌ NO - {e}")

# 2. Can we query positions?
print("\n[TEST 2] Can we query positions?")
try:
    positions = c.futures_position_information()
    open_pos = [p for p in positions if float(p['positionAmt']) != 0]
    print(f"✅ YES - {len(open_pos)} open positions")
except Exception as e:
    print(f"❌ NO - {e}")

# 3. Can we cancel orders?
print("\n[TEST 3] Can we cancel orders?")
try:
    result = c.futures_cancel_all_open_orders(symbol='SOLUSDT')
    print(f"✅ YES - Result: {result}")
except Exception as e:
    error_str = str(e)
    if 'No such open order' in error_str:
        print(f"✅ YES - No orders to cancel (expected)")
    else:
        print(f"❌ NO - {e}")

# 4. Can we change leverage?
print("\n[TEST 4] Can we change leverage?")
try:
    result = c.futures_change_leverage(symbol='SOLUSDT', leverage=20)
    print(f"✅ YES - Leverage changed: {result}")
except Exception as e:
    print(f"⚠️  {e}")

# 5. Can we place a LIMIT order (not stop order)?
print("\n[TEST 5] Can we place a simple LIMIT order?")
try:
    ticker = c.futures_symbol_ticker(symbol='SOLUSDT')
    price = float(ticker['price'])
    
    # Very far from market (won't fill)
    limit_price = round(price * 0.5, 2)  # 50% below market
    
    print(f"   Attempting: BUY LIMIT 1.0 @ ${limit_price} (current: ${price})")
    
    order = c.futures_create_order(
        symbol='SOLUSDT',
        side='BUY',
        type='LIMIT',
        timeInForce='GTC',
        quantity=1.0,
        price=limit_price
    )
    
    print(f"✅ YES - Limit order created: {order['orderId']}")
    
    # Cancel it
    c.futures_cancel_order(symbol='SOLUSDT', orderId=order['orderId'])
    print(f"   Cancelled test order")
    
except Exception as e:
    print(f"❌ NO - {e}")

# 6. Can we place a MARKET order? (dangerous test, small amount)
print("\n[TEST 6] Can we place a tiny MARKET order? (REAL TRADE - 1 SOLUSDT)")
response = input("   WARNING: This will open a real position. Continue? (yes/no): ")

if response.lower() == 'yes':
    try:
        order = c.futures_create_order(
            symbol='SOLUSDT',
            side='BUY',
            type='MARKET',
            quantity=1.0
        )
        
        print(f"✅ YES - Market order executed: {order['orderId']}")
        print(f"   Position opened - you need to close it manually!")
        
    except Exception as e:
        print(f"❌ NO - {e}")
else:
    print("   Skipped (safe choice)")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

print("""
If we CAN place LIMIT orders but NOT STOP orders:
  → The issue is specifically with STOP/TAKE_PROFIT order types
  → This could be a Binance testnet limitation on conditional orders
  
If we CANNOT place ANY orders (including LIMIT):
  → Account trading is disabled or restricted
  → Check API key permissions
  
If we CAN place MARKET orders but NOT STOP orders:
  → Binance has blocked conditional orders on this testnet account
  → Solution: Reset testnet, use different account, or use production
""")
print("=" * 80)

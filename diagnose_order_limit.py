#!/usr/bin/env python3
"""
COMPREHENSIVE ORDER LIMIT DIAGNOSTIC
Finds ROOT CAUSE of -4045 error, not just symptoms
"""

from binance.client import Client
import os
import time
import json
from datetime import datetime

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print("=" * 80)
print("BINANCE ORDER LIMIT ROOT CAUSE ANALYSIS")
print("=" * 80)
print(f"Time: {datetime.now().isoformat()}")
print()

# STEP 1: Get account info
print("[STEP 1] Account Information")
print("-" * 80)
try:
    account = c.futures_account()
    print(f"✓ Account Balance: ${float(account['totalWalletBalance']):.2f}")
    print(f"✓ Can Trade: {account.get('canTrade', 'Unknown')}")
    print(f"✓ Can Deposit: {account.get('canDeposit', 'Unknown')}")
except Exception as e:
    print(f"✗ Account Error: {e}")

# STEP 2: Get ALL orders (including hidden ones)
print("\n[STEP 2] Complete Order Inventory")
print("-" * 80)

all_orders = c.futures_get_open_orders()
print(f"Total visible open orders: {len(all_orders)}")

if all_orders:
    print("\nOrder breakdown:")
    by_type = {}
    by_symbol = {}
    for order in all_orders:
        order_type = order.get('type', 'UNKNOWN')
        symbol = order['symbol']
        by_type[order_type] = by_type.get(order_type, 0) + 1
        by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
    
    for order_type, count in sorted(by_type.items()):
        print(f"  {order_type}: {count}")
    
    print("\nOrders per symbol:")
    for symbol, count in sorted(by_symbol.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {count} orders")
else:
    print("✓ No visible open orders")

# STEP 3: Check positions
print("\n[STEP 3] Position Analysis")
print("-" * 80)

positions = [p for p in c.futures_position_information() if float(p['positionAmt']) != 0]
print(f"Open positions: {len(positions)}")

position_details = []
for pos in positions:
    symbol = pos['symbol']
    amt = float(pos['positionAmt'])
    position_details.append({
        'symbol': symbol,
        'amount': amt,
        'side': 'LONG' if amt > 0 else 'SHORT',
        'entry': float(pos['entryPrice']),
        'mark': float(pos['markPrice']),
        'unrealized_pnl': float(pos['unRealizedProfit'])
    })
    print(f"  {symbol}: {amt:.4f} @ ${pos['entryPrice']}")

# STEP 4: Test order creation on EACH symbol to find which fails
print("\n[STEP 4] Testing Order Creation Per Symbol")
print("-" * 80)
print("This will identify EXACTLY which symbol hits the limit...")

failed_symbols = []
success_count = 0

for i, pos in enumerate(position_details):
    symbol = pos['symbol']
    amt = pos['amount']
    side = 'SELL' if amt > 0 else 'BUY'
    
    # Get proper precision
    try:
        exchange_info = c.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_info:
            print(f"  {i+1}. {symbol}: ✗ Symbol info not found")
            continue
        
        # Get price filter
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        
        # Get lot size
        lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        step_size = float(lot_filter['stepSize']) if lot_filter else 0.01
        
        # Calculate test price with proper precision
        mark_price = pos['mark']
        test_price_multiplier = 1.10 if amt > 0 else 0.90
        test_price = mark_price * test_price_multiplier
        
        # Round to tick size
        test_price = round(test_price / tick_size) * tick_size
        
        # Calculate test quantity with proper precision
        test_qty = abs(amt) * 0.05  # Only 5% for safety
        test_qty = round(test_qty / step_size) * step_size
        
        if test_qty == 0:
            test_qty = step_size
        
        print(f"\n  {i+1}. {symbol} ({pos['side']})")
        print(f"     Testing: {side} TAKE_PROFIT_MARKET")
        print(f"     Price: ${test_price:.8f} (tick={tick_size})")
        print(f"     Qty: {test_qty} (step={step_size})")
        
        # Attempt to create order
        try:
            test_order = c.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=test_price,
                quantity=test_qty,
                workingType='MARK_PRICE',
                positionSide='BOTH'
            )
            
            order_id = test_order['orderId']
            print(f"     ✓ Order created: {order_id}")
            
            # Cancel immediately
            c.futures_cancel_order(symbol=symbol, orderId=order_id)
            print(f"     ✓ Order cancelled")
            success_count += 1
            
            time.sleep(0.3)  # Small delay
            
        except Exception as e:
            error_str = str(e)
            print(f"     ✗ FAILED: {error_str}")
            
            if '-4045' in error_str:
                print(f"     ⚠️  ROOT CAUSE FOUND: {symbol} triggers -4045!")
                failed_symbols.append(symbol)
                
                # Get detailed info about this symbol
                print(f"     Investigating {symbol}...")
                
                # Check for existing orders
                symbol_orders = c.futures_get_open_orders(symbol=symbol)
                print(f"     Visible orders for {symbol}: {len(symbol_orders)}")
                
                if symbol_orders:
                    for order in symbol_orders:
                        print(f"       - {order['type']}: {order['origQty']} @ {order.get('stopPrice', order.get('price', 'N/A'))}")
                
                # Check position risk
                try:
                    pos_risk = c.futures_position_information(symbol=symbol)
                    for pr in pos_risk:
                        if float(pr['positionAmt']) != 0:
                            print(f"     Position Risk Info:")
                            print(f"       Leverage: {pr.get('leverage', 'N/A')}")
                            print(f"       Position: {pr['positionAmt']}")
                            print(f"       Entry: {pr['entryPrice']}")
                except Exception as pr_e:
                    print(f"     Could not get position risk: {pr_e}")
            
            elif '-1111' in error_str:
                print(f"     ⚠️  Precision error (not the root cause)")
            else:
                print(f"     ⚠️  Other error: {error_str}")
                failed_symbols.append(symbol)
    
    except Exception as ex:
        print(f"  {i+1}. {symbol}: ✗ Exception: {ex}")
        failed_symbols.append(symbol)

# STEP 5: Summary and recommendations
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\nPositions tested: {len(position_details)}")
print(f"Successful tests: {success_count}")
print(f"Failed tests: {len(failed_symbols)}")

if failed_symbols:
    print(f"\n⚠️  PROBLEM SYMBOLS:")
    for sym in failed_symbols:
        print(f"  - {sym}")
    
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    print(f"   These {len(failed_symbols)} symbols are hitting order limits")
    print(f"   Possible causes:")
    print(f"   1. Hidden/stuck orders not shown in futures_get_open_orders()")
    print(f"   2. Recently cancelled orders still counting toward limit")
    print(f"   3. Symbol-specific limit lower than expected")
    print(f"   4. Binance testnet issue (try production)")
    
    print(f"\n💡 RECOMMENDED FIXES:")
    print(f"   1. Wait 60 seconds after cancelling orders")
    print(f"   2. Use simpler TP/SL structure (1 order each instead of partial+trailing)")
    print(f"   3. Cancel ALL orders for problem symbols before setting new ones")
    print(f"   4. Use production API instead of testnet")
else:
    print(f"\n✅ All symbols passed! No -4045 errors detected.")
    print(f"   The issue may be timing-related or intermittent")

# STEP 6: Check if timing is the issue
print("\n[STEP 6] Rapid Creation Test (timing issue check)")
print("-" * 80)

if len(position_details) > 0:
    test_symbol = position_details[0]['symbol']
    print(f"Testing rapid order creation on {test_symbol}...")
    
    rapid_created = []
    for attempt in range(5):
        try:
            pos = position_details[0]
            amt = pos['amount']
            side = 'SELL' if amt > 0 else 'BUY'
            test_price = pos['mark'] * (1.10 if amt > 0 else 0.90)
            
            order = c.futures_create_order(
                symbol=test_symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=test_price,
                quantity=abs(amt) * 0.05,
                workingType='MARK_PRICE',
                positionSide='BOTH'
            )
            rapid_created.append(order['orderId'])
            print(f"  Attempt {attempt+1}: ✓ Created order {order['orderId']}")
        except Exception as e:
            print(f"  Attempt {attempt+1}: ✗ Failed - {e}")
            if '-4045' in str(e):
                print(f"  ⚠️  Hit limit after {len(rapid_created)} rapid orders!")
                break
    
    # Clean up
    for order_id in rapid_created:
        try:
            c.futures_cancel_order(symbol=test_symbol, orderId=order_id)
        except:
            pass
    
    print(f"  Created {len(rapid_created)} orders before hitting limit (if any)")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

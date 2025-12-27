#!/usr/bin/env python3
"""
Test Stop Loss and Take Profit Execution
Places orders with SL/TP and monitors their execution
"""

import sys
import os
from datetime import datetime
import time

sys.path.insert(0, '/app/backend')

from binance.client import Client

print("\n" + "="*70)
print("🛡️  STOP LOSS & TAKE PROFIT TEST")
print("="*70)
print(f"⏰ Timestamp: {datetime.utcnow().isoformat()}\n")

# Initialize client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# =============================================================================
# STEP 1: Get current position and price
# =============================================================================
print("📊 STEP 1: Current Position Analysis")
print("-" * 70)

symbol = 'BTCUSDT'
current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
print(f"Current Price: ${current_price:,.2f}")

positions = client.futures_position_information(symbol=symbol)
for pos in positions:
    if float(pos['positionAmt']) != 0:
        position_amt = float(pos['positionAmt'])
        entry_price = float(pos['entryPrice'])
        unrealized_pnl = float(pos['unRealizedProfit'])
        
        print(f"\n📈 Current Position:")
        print(f"   Amount: {position_amt} BTC")
        print(f"   Entry: ${entry_price:,.2f}")
        print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
        
        # Calculate stop loss and take profit prices
        if position_amt > 0:  # LONG position
            side_text = "LONG"
            sl_price = entry_price * 0.99  # 1% below entry
            tp_price = entry_price * 1.01  # 1% above entry
            sl_side = 'SELL'
            tp_side = 'SELL'
        else:  # SHORT position
            side_text = "SHORT"
            sl_price = entry_price * 1.01  # 1% above entry
            tp_price = entry_price * 0.99  # 1% below entry
            sl_side = 'BUY'
            tp_side = 'BUY'
        
        print(f"   Side: {side_text}")
        print(f"\n🛡️  Calculated Levels:")
        print(f"   Stop Loss: ${sl_price:,.2f} ({sl_side})")
        print(f"   Take Profit: ${tp_price:,.2f} ({tp_side})")

# =============================================================================
# STEP 2: Place Stop Loss Order
# =============================================================================
print("\n" + "="*70)
print("🛑 STEP 2: Placing Stop Loss Order")
print("-" * 70)

try:
    # Get open orders first
    open_orders = client.futures_get_open_orders(symbol=symbol)
    print(f"Current open orders: {len(open_orders)}")
    
    # Check if we have a position to protect
    if position_amt != 0:
        quantity = abs(position_amt)
        
        print(f"\n📋 Stop Loss Order Details:")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {sl_side}")
        print(f"   Type: STOP_MARKET")
        print(f"   Stop Price: ${sl_price:,.2f}")
        print(f"   Quantity: {quantity} BTC")
        print(f"\n⚠️  Placing STOP_MARKET order...")
        
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type='STOP_MARKET',
            stopPrice=str(round(sl_price, 2)),
            quantity=str(quantity),
            closePosition=True
        )
        
        print(f"✅ Stop Loss Order Placed!")
        print(f"   Order ID: {sl_order['orderId']}")
        print(f"   Status: {sl_order['status']}")
        
    else:
        print("⚠️  No position found to place stop loss")
        
except Exception as e:
    print(f"❌ Failed to place stop loss: {e}")

# =============================================================================
# STEP 3: Place Take Profit Order
# =============================================================================
print("\n" + "="*70)
print("💰 STEP 3: Placing Take Profit Order")
print("-" * 70)

try:
    if position_amt != 0:
        quantity = abs(position_amt)
        
        print(f"\n📋 Take Profit Order Details:")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {tp_side}")
        print(f"   Type: TAKE_PROFIT_MARKET")
        print(f"   Stop Price: ${tp_price:,.2f}")
        print(f"   Quantity: {quantity} BTC")
        print(f"\n⚠️  Placing TAKE_PROFIT_MARKET order...")
        
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=str(round(tp_price, 2)),
            quantity=str(quantity),
            closePosition=True
        )
        
        print(f"✅ Take Profit Order Placed!")
        print(f"   Order ID: {tp_order['orderId']}")
        print(f"   Status: {tp_order['status']}")
        
    else:
        print("⚠️  No position found to place take profit")
        
except Exception as e:
    print(f"❌ Failed to place take profit: {e}")

# =============================================================================
# STEP 4: Monitor Orders
# =============================================================================
print("\n" + "="*70)
print("👁️  STEP 4: Monitoring Orders (30 seconds)")
print("-" * 70)

for i in range(6):
    time.sleep(5)
    
    # Check current price
    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    
    # Check open orders
    open_orders = client.futures_get_open_orders(symbol=symbol)
    
    # Check position
    positions = client.futures_position_information(symbol=symbol)
    pos_amt = 0
    for pos in positions:
        if float(pos['positionAmt']) != 0:
            pos_amt = float(pos['positionAmt'])
            unrealized_pnl = float(pos['unRealizedProfit'])
    
    print(f"\n⏰ Check {i+1}/6 (T+{(i+1)*5}s):")
    print(f"   Price: ${current_price:,.2f}")
    print(f"   Position: {pos_amt} BTC")
    if pos_amt != 0:
        print(f"   P&L: ${unrealized_pnl:.2f}")
    print(f"   Open Orders: {len(open_orders)}")
    
    # Display open orders
    if open_orders:
        for order in open_orders:
            print(f"      - {order['type']}: {order['side']} @ ${float(order['stopPrice']):,.2f}")
    
    # Check if position closed
    if pos_amt == 0 and position_amt != 0:
        print("\n🎯 POSITION CLOSED!")
        print("   Checking which order triggered...")
        
        # Get recent trades
        recent_orders = client.futures_get_all_orders(symbol=symbol, limit=5)
        for order in recent_orders:
            if order['status'] == 'FILLED' and order['orderId'] in [sl_order.get('orderId'), tp_order.get('orderId')]:
                if order['orderId'] == sl_order.get('orderId'):
                    print(f"   🛑 STOP LOSS TRIGGERED at ${float(order['avgPrice']):,.2f}")
                elif order['orderId'] == tp_order.get('orderId'):
                    print(f"   💰 TAKE PROFIT TRIGGERED at ${float(order['avgPrice']):,.2f}")
        break

# =============================================================================
# FINAL STATUS
# =============================================================================
print("\n" + "="*70)
print("📊 FINAL STATUS")
print("="*70)

# Get final position
final_positions = client.futures_position_information(symbol=symbol)
final_pos_amt = 0
for pos in final_positions:
    if float(pos['positionAmt']) != 0:
        final_pos_amt = float(pos['positionAmt'])

# Get open orders
final_open_orders = client.futures_get_open_orders(symbol=symbol)

print(f"\n✅ Test Complete!")
print(f"   Final Position: {final_pos_amt} BTC")
print(f"   Open Orders: {len(final_open_orders)}")

if final_open_orders:
    print(f"\n   📋 Remaining Open Orders:")
    for order in final_open_orders:
        print(f"      - Order {order['orderId']}: {order['type']} {order['side']} @ ${float(order['stopPrice']):,.2f}")
    print(f"\n   💡 Note: Orders will remain active until triggered or cancelled")
else:
    print(f"\n   ✅ All orders executed or no orders remain")

print("\n" + "="*70 + "\n")

print("💡 SUMMARY:")
print("   - Stop Loss orders protect against downside")
print("   - Take Profit orders lock in gains")
print("   - Both are STOP_MARKET/TAKE_PROFIT_MARKET types")
print("   - Orders remain active until price triggers them\n")

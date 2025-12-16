#!/usr/bin/env python3
"""Check OPUSDT orders and position state"""

from binance.client import Client
import os
import json
from pathlib import Path

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Get current position
positions = client.futures_position_information(symbol='OPUSDT')
pos = [p for p in positions if float(p['positionAmt']) != 0][0]

print("="*70)
print("OPUSDT POSITION & ORDERS ANALYSIS")
print("="*70)

print("\n📊 CURRENT POSITION:")
print(f"   Size: {pos['positionAmt']} OP")
print(f"   Entry: ${pos['entryPrice']}")
print(f"   Mark: ${pos['markPrice']}")
print(f"   PnL: ${pos['unRealizedProfit']} ({float(pos['unRealizedProfit'])/abs(float(pos['positionAmt'])*float(pos['entryPrice']))*100:.2f}%)")

# Get OPUSDT orders
orders = client.futures_get_open_orders(symbol='OPUSDT')

print(f"\n📋 OPEN ORDERS: {len(orders)}")
if not orders:
    print("   ❌ NO ORDERS FOUND - Position is UNPROTECTED!")
else:
    for i, order in enumerate(orders, 1):
        print(f"\n   Order {i}: {order['type']}")
        print(f"      Side: {order['side']}")
        print(f"      Quantity: {order['origQty']}")
        print(f"      Price: {order.get('price', 'N/A')}")
        print(f"      Stop Price: {order.get('stopPrice', 'N/A')}")
        print(f"      Activate Price: {order.get('activatePrice', 'N/A')}")
        print(f"      Callback Rate: {order.get('priceRate', 'N/A')}")

# Check position state file
state_file = Path('/app/backend/data/position_state.json')
print(f"\n📁 POSITION STATE FILE:")
if state_file.exists():
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    if 'OPUSDT' in state:
        opusdt_state = state['OPUSDT']
        print(f"   trail_percentage: {opusdt_state.get('trail_percentage', '❌ NOT SET')}")
        print(f"   initial_stop_loss: {opusdt_state.get('initial_stop_loss', 'NOT SET')}")
        print(f"   take_profit: {opusdt_state.get('take_profit', 'NOT SET')}")
        print(f"   highest_profit_pct: {opusdt_state.get('highest_profit_pct', 0)}")
        print(f"   partial_tp_1_hit: {opusdt_state.get('partial_tp_1_hit', False)}")
        print(f"   partial_tp_2_hit: {opusdt_state.get('partial_tp_2_hit', False)}")
    else:
        print("   ❌ OPUSDT not in position state file!")
else:
    print("   ❌ Position state file does not exist!")

print("\n" + "="*70)
print("🔍 DIAGNOSIS:")
print("="*70)

# Calculate what partial TPs should be
entry_price = float(pos['entryPrice'])
mark_price = float(pos['markPrice'])
position_size = abs(float(pos['positionAmt']))
unrealized_pnl_pct = (float(pos['unRealizedProfit']) / (position_size * entry_price)) * 100

print(f"\n💰 PROFIT STATUS:")
print(f"   Current PnL: {unrealized_pnl_pct:.2f}%")
print(f"   Mark Price: ${mark_price:.4f}")
print(f"   Entry: ${entry_price:.4f}")

# For SHORT: profit when price goes down
is_short = float(pos['positionAmt']) < 0
if is_short:
    print(f"   Direction: SHORT (profit when price ⬇)")
    
    # Partial TP levels for SHORT
    # Position Monitor says: partial TPs (25%+25%)
    # RL Agent says: partial@1.63%
    
    partial_tp_1_pct = 1.63  # 1.63% profit
    partial_tp_2_pct = 3.25  # 3.25% profit (full TP)
    
    # For SHORT: TP price is BELOW entry
    partial_tp_1_price = entry_price * (1 - partial_tp_1_pct / 100)
    partial_tp_2_price = entry_price * (1 - partial_tp_2_pct / 100)
    
    print(f"\n🎯 EXPECTED PARTIAL TPs:")
    print(f"   25% at {partial_tp_1_pct:.2f}% profit → ${partial_tp_1_price:.4f}")
    print(f"   50% at {partial_tp_2_pct:.2f}% profit → ${partial_tp_2_price:.4f}")
    
    if mark_price <= partial_tp_1_price:
        print(f"   ✅ Price reached 1st partial TP level!")
        if not any(o['type'] == 'TAKE_PROFIT_MARKET' for o in orders):
            print(f"   ❌ BUT NO PARTIAL TP ORDER EXISTS!")
    else:
        distance_to_tp1 = ((mark_price - partial_tp_1_price) / mark_price) * 100
        print(f"   ⏳ Price needs to drop {distance_to_tp1:.2f}% more to hit 1st TP")

print("\n🔧 RECOMMENDED ACTIONS:")
if len(orders) == 0:
    print("   1. ❌ CRITICAL: Position has NO protective orders!")
    print("   2. Position Monitor is trying to set TP/SL but failing")
    print("   3. Check logs for 'OPUSDT UNPROTECTED' errors")
elif 'trail_percentage' not in opusdt_state if 'OPUSDT' in state else True:
    print("   1. ⚠️ trail_percentage NOT SET in position state")
    print("   2. Trailing Stop Manager cannot manage this position")
    print("   3. Partial TPs may not trigger automatically")

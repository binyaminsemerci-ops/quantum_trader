#!/usr/bin/env python3
"""Check current positions and their stop loss protection"""
import os
from binance.um_futures import UMFutures

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = UMFutures(key=api_key, secret=api_secret)

print("\n" + "="*80)
print("[SEARCH] CHECKING LIVE POSITIONS & STOP LOSS ORDERS")
print("="*80)

# Get positions
positions = client.get_position_risk()
active_positions = [p for p in positions if float(p['positionAmt']) != 0]

print(f"\n[CHART] Active Positions: {len(active_positions)}\n")

for pos in active_positions:
    symbol = pos['symbol']
    amt = float(pos['positionAmt'])
    entry = float(pos['entryPrice'])
    mark = float(pos['markPrice'])
    pnl = float(pos['unRealizedProfit'])
    leverage = pos['leverage']
    
    side = "LONG" if amt > 0 else "SHORT"
    pnl_pct = (pnl / (abs(amt) * entry)) * 100 * float(leverage)
    
    print(f"{'[GREEN_CIRCLE]' if amt > 0 else '[RED_CIRCLE]'} {side} {symbol}")
    print(f"   Amount: {abs(amt)}")
    print(f"   Entry: ${entry:.5f}")
    print(f"   Mark: ${mark:.5f}")
    print(f"   Leverage: {leverage}x")
    print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    
    # Calculate expected SL price (2% for LONG, -2% for SHORT)
    if amt > 0:  # LONG
        expected_sl = entry * 0.98  # -2%
    else:  # SHORT
        expected_sl = entry * 1.02  # +2%
    
    print(f"   Expected SL: ${expected_sl:.5f} (2% protection)")
    
    # Check for stop loss orders
    try:
        orders = client.get_open_orders(symbol=symbol)
        sl_orders = [o for o in orders if 'STOP' in o['type']]
        
        if sl_orders:
            print(f"   [OK] Stop Loss Orders Found: {len(sl_orders)}")
            for order in sl_orders:
                order_type = order['type']
                stop_price = float(order['stopPrice'])
                price = order.get('price', 'MARKET')
                
                print(f"      • Type: {order_type}")
                print(f"      • Stop Price: ${stop_price:.5f}")
                print(f"      • Limit Price: ${price}")
                print(f"      • Time In Force: {order.get('timeInForce', 'N/A')}")
                
                # Verify it's STOP_LOSS not STOP_MARKET
                if order_type == 'STOP':
                    print(f"      [WARNING]  WARNING: This is STOP_MARKET (old type - can fail!)")
                elif order_type == 'STOP_MARKET':
                    print(f"      [WARNING]  WARNING: STOP_MARKET type (can fail in volatile markets!)")
                elif 'STOP' in order_type and price != 'MARKET':
                    print(f"      [OK] STOP_LOSS with limit price (GUARANTEED EXECUTION)")
                
        else:
            print(f"   [ALERT] NO STOP LOSS ORDERS FOUND!")
            print(f"      Position is UNPROTECTED - manual intervention needed!")
            
    except Exception as e:
        print(f"   ❌ Error checking orders: {e}")
    
    print()

print("="*80)

# Summary
if active_positions:
    total_pnl = sum(float(p['unRealizedProfit']) for p in active_positions)
    print(f"\n[MONEY] Total P&L: ${total_pnl:.2f}\n")
else:
    print("\n[OK] No active positions\n")

print("="*80 + "\n")

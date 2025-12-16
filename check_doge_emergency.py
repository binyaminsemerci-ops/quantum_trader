#!/usr/bin/env python3
"""Emergency check for DOGEUSDT position and stop loss."""
import os
from binance.client import Client

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))

# Get position
positions = client.futures_position_information()
doge_pos = [p for p in positions if p['symbol'] == 'DOGEUSDT' and float(p['positionAmt']) != 0]

if not doge_pos:
    print("❌ No DOGEUSDT position found")
    exit(1)

pos = doge_pos[0]
entry_price = float(pos['entryPrice'])
mark_price = float(pos['markPrice'])
position_amt = float(pos['positionAmt'])
unrealized_pnl = float(pos['unRealizedProfit'])
liquidation = float(pos['liquidationPrice'])

print(f"\n[SEARCH] DOGEUSDT POSITION ANALYSIS")
print(f"{'='*60}")
print(f"Position Size: {position_amt:,.0f} DOGE (SHORT)")
print(f"Entry Price: ${entry_price:.6f}")
print(f"Mark Price: ${mark_price:.6f}")
print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
print(f"Liquidation: ${liquidation:.6f}")

# Calculate loss percentage
loss_pct = ((mark_price - entry_price) / entry_price) * 100
print(f"\n📉 Loss: {loss_pct:.2f}%")

# Check stop loss orders
orders = client.futures_get_open_orders(symbol='DOGEUSDT')
stop_orders = [o for o in orders if o['type'] == 'STOP_MARKET' and o.get('closePosition')]

print(f"\n[WARNING]  STOP_MARKET ORDERS:")
if stop_orders:
    for order in stop_orders:
        stop_price = float(order['stopPrice'])
        print(f"  Stop Price: ${stop_price:.6f}")
        print(f"  Working Type: {order.get('workingType', 'N/A')}")
        print(f"  Status: {order['status']}")
        
        # Calculate distance
        if position_amt < 0:  # SHORT
            distance_pct = ((stop_price - mark_price) / mark_price) * 100
            print(f"  Distance to trigger: {distance_pct:.2f}% above current price")
            
            if mark_price >= stop_price:
                print(f"\n  [ALERT] CRITICAL: Mark price ABOVE stop! Should have triggered!")
            else:
                print(f"  [OK] Stop will trigger at ${stop_price:.6f}")
else:
    print("  ❌ NO STOP_MARKET ORDER FOUND!")

# Show all orders
print(f"\n[CLIPBOARD] ALL ORDERS ({len(orders)}):")
for order in orders:
    print(f"  • {order['type']} - Stop: ${float(order.get('stopPrice', 0)):.6f} - Status: {order['status']}")

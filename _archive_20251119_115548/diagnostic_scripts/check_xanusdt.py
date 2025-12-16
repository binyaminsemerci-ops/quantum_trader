#!/usr/bin/env python3
import os
from binance.client import Client

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

# Get position
pos = client.futures_position_information(symbol='XANUSDT')[0]
size = float(pos['positionAmt'])
entry = float(pos['entryPrice'])
mark = float(pos['markPrice'])
pnl = float(pos['unRealizedProfit'])

print(f"[CHART] XANUSDT LONG Position:")
print(f"  Size: {size:,.0f} XAN")
print(f"  Entry: ${entry:.8f}")
print(f"  Current: ${mark:.8f}")
print(f"  PNL: ${pnl:+.2f} ({(pnl/(size*entry))*100:+.2f}%)")

# Get open orders
orders = client.futures_get_open_orders(symbol='XANUSDT')
print(f"\n[SHIELD]  Open Orders: {len(orders)}")

tp_orders = [o for o in orders if o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']]
sl_orders = [o for o in orders if o['type'] in ['STOP_MARKET', 'STOP_LOSS']]

if tp_orders:
    for o in tp_orders:
        print(f"  [OK] TP: ${float(o.get('stopPrice', 0)):.8f} (Order {o['orderId']})")

if sl_orders:
    for o in sl_orders:
        print(f"  [SHIELD]  SL: ${float(o.get('stopPrice', 0)):.8f} (Order {o['orderId']})")

# Check trade state for trailing info
import json
with open('/app/backend/data/trade_state.json', 'r') as f:
    trade_state = json.load(f)

if 'XANUSDT' in trade_state:
    state = trade_state['XANUSDT']
    print(f"\n[TARGET] Trade State:")
    print(f"  Entry: ${state.get('avg_entry', 0):.8f}")
    print(f"  Peak: ${state.get('peak', 0):.8f}")
    
    if 'ai_trail_pct' in state:
        trail_pct = state['ai_trail_pct']
        peak = state.get('peak', entry)
        trail_sl = peak * (1 - trail_pct)
        print(f"\n🔄 Trailing Stop:")
        print(f"  Trail %: {trail_pct*100:.1f}%")
        print(f"  Peak: ${peak:.8f}")
        print(f"  Trail SL would be: ${trail_sl:.8f}")
        
        # Check if trailing is active
        profit_pct = (mark - entry) / entry
        if profit_pct >= 0.005:
            print(f"  [OK] Trailing ACTIVE (profit: {profit_pct*100:+.2f}%)")
        else:
            print(f"  ⏳ Trailing waiting (need 0.5% profit, current: {profit_pct*100:+.2f}%)")

#!/usr/bin/env python3
"""Get detailed position and TP/SL information"""
from binance.client import Client
import os

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret, testnet=True)

print("=" * 80)
print("POSITION AND TP/SL OVERVIEW")
print("=" * 80)

# Get all positions
all_positions = client.futures_position_information()
active_positions = [p for p in all_positions if float(p.get('positionAmt', 0)) != 0]

print(f"\nFound {len(active_positions)} active positions\n")

for pos in sorted(active_positions, key=lambda x: float(x.get('unRealizedProfit', 0))):
    symbol = pos['symbol']
    amt = float(pos['positionAmt'])
    entry = float(pos['entryPrice'])
    mark = float(pos['markPrice'])
    pnl = float(pos['unRealizedProfit'])
    pnl_pct = (float(mark) - float(entry)) / float(entry) * 100 if amt > 0 else (float(entry) - float(mark)) / float(entry) * 100
    
    side = "LONG" if amt > 0 else "SHORT"
    position_value = abs(amt) * mark
    
    print(f"\n{'='*80}")
    print(f"📊 {symbol} ({side})")
    print(f"   Amount: {amt}")
    print(f"   Entry: ${entry:.8f}")
    print(f"   Mark: ${mark:.8f}")
    print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print(f"   Position Value: ${position_value:.2f}")
    
    # Get open orders for this symbol
    orders = client.futures_get_open_orders(symbol=symbol)
    
    tp_orders = [o for o in orders if o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']]
    sl_orders = [o for o in orders if o['type'] in ['STOP_MARKET', 'STOP_LOSS']]
    
    if tp_orders:
        for tp in tp_orders:
            tp_price = float(tp.get('stopPrice', 0))
            tp_distance = abs(tp_price - mark) / mark * 100
            tp_profit = (tp_price - entry) / entry * 100 if amt > 0 else (entry - tp_price) / entry * 100
            print(f"   ✅ TP: ${tp_price:.8f} (Distance: {tp_distance:.2f}%, Profit: {tp_profit:+.2f}%)")
    else:
        print(f"   ❌ NO TAKE PROFIT SET!")
    
    if sl_orders:
        for sl in sl_orders:
            sl_price = float(sl.get('stopPrice', 0))
            sl_distance = abs(sl_price - mark) / mark * 100
            sl_loss = (sl_price - entry) / entry * 100 if amt > 0 else (entry - sl_price) / entry * 100
            print(f"   🛡️ SL: ${sl_price:.8f} (Distance: {sl_distance:.2f}%, Loss: {sl_loss:+.2f}%)")
    else:
        print(f"   ⚠️ NO STOP LOSS SET!")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total Positions: {len(active_positions)}")
print(f"With TP: {sum(1 for pos in active_positions if any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in client.futures_get_open_orders(symbol=pos['symbol'])))}")
print(f"With SL: {sum(1 for pos in active_positions if any(o['type'] in ['STOP_MARKET', 'STOP_LOSS'] for o in client.futures_get_open_orders(symbol=pos['symbol'])))}")

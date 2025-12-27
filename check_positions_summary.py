#!/usr/bin/env python3
"""Quick position summary."""
import os
import sys
sys.path.insert(0, '/app')
from binance.client import Client

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

positions = client.futures_position_information()
open_pos = [p for p in positions if float(p['positionAmt']) != 0]

print(f"\n[CHART] ÅPNE POSISJONER: {len(open_pos)}\n")

for pos in open_pos:
    symbol = pos['symbol']
    amt = float(pos['positionAmt'])
    entry = float(pos['entryPrice'])
    mark = float(pos['markPrice'])
    pnl = float(pos['unRealizedProfit'])
    leverage = float(pos.get('leverage', 20))  # Default to 20x if not available
    
    side = "LONG" if amt > 0 else "SHORT"
    pnl_pct = (pnl / (abs(amt) * entry / leverage)) * 100
    
    print(f"🔹 {symbol} {side}")
    print(f"   Size: {abs(amt):.4f} | Leverage: {leverage}x")
    print(f"   Entry: ${entry:.6f} | Mark: ${mark:.6f}")
    print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print()

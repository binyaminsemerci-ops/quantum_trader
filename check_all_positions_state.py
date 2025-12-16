#!/usr/bin/env python3
"""Check if ALL open positions have proper trail/TP/SL settings"""

import json
from pathlib import Path
from binance.client import Client
import os

# Get current open positions from Binance
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)
positions = client.futures_position_information()
open_positions = [p for p in positions if float(p['positionAmt']) != 0]

# Load trade state
state_file = Path('/app/backend/data/trade_state.json')
with open(state_file, 'r') as f:
    state = json.load(f)

print("="*70)
print("CHECKING ALL OPEN POSITIONS FOR TRAIL/TP/SL SETTINGS")
print("="*70)

missing_count = 0
correct_count = 0

for pos in open_positions:
    symbol = pos['symbol']
    size = float(pos['positionAmt'])
    entry = float(pos['entryPrice'])
    pnl = float(pos['unRealizedProfit'])
    
    print(f"\n{'='*70}")
    print(f"📊 {symbol}: {abs(size):.2f} @ ${entry:.4f} (PnL: ${pnl:.2f})")
    print(f"{'='*70}")
    
    # Check if symbol in state
    if symbol not in state:
        print(f"   ❌ IKKE I TRADE STATE!")
        missing_count += 1
        continue
    
    sym_state = state[symbol]
    
    # Check critical fields
    has_trail = 'ai_trail_pct' in sym_state
    has_tp = 'ai_tp_pct' in sym_state
    has_sl = 'ai_sl_pct' in sym_state
    has_partial = 'ai_partial_tp' in sym_state
    
    print(f"\n   🔍 STATE FIELDS:")
    print(f"      ai_trail_pct: {'✅ ' + str(sym_state.get('ai_trail_pct')) if has_trail else '❌ MANGLER'}")
    print(f"      ai_tp_pct: {'✅ ' + str(sym_state.get('ai_tp_pct')) if has_tp else '❌ MANGLER'}")
    print(f"      ai_sl_pct: {'✅ ' + str(sym_state.get('ai_sl_pct')) if has_sl else '❌ MANGLER'}")
    print(f"      ai_partial_tp: {'✅ ' + str(sym_state.get('ai_partial_tp')) if has_partial else '❌ MANGLER'}")
    
    # Check if position data is current
    state_qty = sym_state.get('qty', 0)
    state_entry = sym_state.get('avg_entry', 0)
    
    qty_match = abs(abs(state_qty) - abs(size)) < 1  # Allow 1 unit difference
    entry_match = abs(state_entry - entry) < 0.01  # Allow 1% difference
    
    print(f"\n   📦 DATA SYNC:")
    print(f"      Quantity: {'✅ Match' if qty_match else f'❌ State={state_qty}, Actual={size}'}")
    print(f"      Entry: {'✅ Match' if entry_match else f'❌ State={state_entry}, Actual={entry}'}")
    
    # Overall status
    if has_trail and has_tp and has_sl and qty_match and entry_match:
        print(f"\n   ✅ STATUS: KORREKT - Trailing stops vil fungere!")
        correct_count += 1
    else:
        print(f"\n   ❌ STATUS: FEIL - Trailing stops vil IKKE fungere!")
        missing_count += 1

print(f"\n{'='*70}")
print(f"📊 SAMMENDRAG:")
print(f"{'='*70}")
print(f"   Totalt posisjoner: {len(open_positions)}")
print(f"   ✅ Korrekt konfigurert: {correct_count}")
print(f"   ❌ Mangler innstillinger: {missing_count}")

if missing_count > 0:
    print(f"\n   ⚠️ KRITISK: {missing_count} posisjoner vil IKKE få partial profit!")
    print(f"   🔧 Løsning: Kjør fix script for hver posisjon")
else:
    print(f"\n   ✅ PERFEKT: Alle posisjoner er korrekt konfigurert!")

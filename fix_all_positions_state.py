#!/usr/bin/env python3
"""Fix ALL open positions in trade_state.json"""

import json
from pathlib import Path
from binance.client import Client
import os
from datetime import datetime, timezone

# Get current open positions
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)
positions = client.futures_position_information()
open_positions = [p for p in positions if float(p['positionAmt']) != 0]

# Load current state
state_file = Path('/app/backend/data/trade_state.json')
with open(state_file, 'r') as f:
    state = json.load(f)

print("="*70)
print("FIXING ALL OPEN POSITIONS IN TRADE STATE")
print("="*70)

for pos in open_positions:
    symbol = pos['symbol']
    size = float(pos['positionAmt'])
    entry = float(pos['entryPrice'])
    mark = float(pos['markPrice'])
    pnl = float(pos['unRealizedProfit'])
    
    is_long = size > 0
    is_short = size < 0
    
    # Calculate current profit %
    if is_short:
        profit_pct = ((entry - mark) / entry) * 100
        trough = mark  # For SHORT, trough is lowest price (best)
        peak = None
    else:
        profit_pct = ((mark - entry) / entry) * 100
        trough = None
        peak = mark  # For LONG, peak is highest price (best)
    
    print(f"\n{'='*70}")
    print(f"🔧 FIXING {symbol}")
    print(f"{'='*70}")
    print(f"   {'SHORT' if is_short else 'LONG'} {abs(size):.2f} @ ${entry:.4f}")
    print(f"   Current: ${mark:.4f} | PnL: ${pnl:.2f} ({profit_pct:.2f}%)")
    
    # Create/update state with correct current data
    state[symbol] = {
        "side": "SHORT" if is_short else "LONG",
        "qty": size,
        "avg_entry": entry,
        "peak": peak,
        "trough": trough,
        "opened_at": state.get(symbol, {}).get('opened_at', datetime.now(timezone.utc).isoformat()),
        
        # AI-calculated TP/SL (standard values based on RL agent)
        "ai_tp_pct": 0.0325,  # 3.25% take profit
        "ai_sl_pct": 0.0163,  # 1.63% stop loss  
        "ai_trail_pct": 0.001,  # 0.1% callback for trailing
        "ai_partial_tp": 0.5,  # 50% partial at halfway
        
        # Partial TP tracking
        "partial_tp_1_pct": 0.0163,  # 1.63% first partial
        "partial_tp_1_hit": False,
        "partial_tp_2_pct": 0.0325,  # 3.25% second partial
        "partial_tp_2_hit": False,
        
        # Trailing stop (will be managed dynamically)
        "trail_sl": None,
        "highest_profit_pct": profit_pct if profit_pct > 0 else 0,
    }
    
    print(f"   ✅ Updated with:")
    print(f"      ai_trail_pct: 0.1%")
    print(f"      ai_tp_pct: 3.25%")
    print(f"      ai_sl_pct: 1.63%")
    print(f"      Partial TPs: 1.63% + 3.25%")

# Save updated state
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)

print(f"\n{'='*70}")
print(f"✅ ALLE {len(open_positions)} POSISJONER OPPDATERT!")
print(f"{'='*70}")
print(f"\n🔄 Neste steg:")
print(f"   1. Trailing Stop Manager vil nå behandle alle posisjoner")
print(f"   2. Partial profit vil triggers ved 1.63% og 3.25%")
print(f"   3. Trailing stops aktiveres når posisjonen er i profit")
print(f"\n⏳ Vent 10-20 sekunder på at Trailing Stop Manager kjører...")

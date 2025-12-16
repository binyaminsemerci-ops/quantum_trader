#!/usr/bin/env python3
"""Fix OPUSDT trade state - add missing trail and TP/SL parameters"""

import json
from pathlib import Path
from datetime import datetime, timezone

state_file = Path('/app/backend/data/trade_state.json')

# Load current state
with open(state_file, 'r') as f:
    state = json.load(f)

print("="*70)
print("FIXING OPUSDT TRADE STATE")
print("="*70)

print("\n📋 CURRENT STATE:")
print(json.dumps(state.get('OPUSDT', {}), indent=2))

# Update OPUSDT with correct current position
state['OPUSDT'] = {
    "side": "SHORT",
    "qty": 57043.1,  # Current position size
    "avg_entry": 0.3252,  # Current entry price
    "peak": None,  # For SHORT, peak is highest price (bad)
    "trough": 0.3237,  # For SHORT, trough is lowest price (good) - current mark
    "opened_at": "2025-12-08T10:19:40+00:00",  # When position was opened
    
    # AI-calculated TP/SL from RL Agent
    "ai_tp_pct": 0.0325,  # 3.25% take profit
    "ai_sl_pct": 0.0163,  # 1.63% stop loss
    "ai_trail_pct": 0.001,  # 0.1% callback for trailing stop
    "ai_partial_tp": 0.5,  # 50% partial take profit
    
    # Partial TP tracking
    "partial_tp_1_pct": 0.0163,  # First partial at 1.63%
    "partial_tp_1_hit": False,
    "partial_tp_2_pct": 0.0325,  # Second partial at 3.25%
    "partial_tp_2_hit": False,
    
    # Trailing stop price (will be updated dynamically)
    "trail_sl": None,  # Not yet activated
    "highest_profit_pct": 0.47,  # Current profit %
}

print("\n✅ NEW STATE:")
print(json.dumps(state['OPUSDT'], indent=2))

# Save updated state
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)

print("\n💾 State file updated!")
print("="*70)
print("🔄 Trailing Stop Manager should now manage OPUSDT")
print("📈 Partial TPs will trigger at 1.63% and 3.25% profit")
print("="*70)

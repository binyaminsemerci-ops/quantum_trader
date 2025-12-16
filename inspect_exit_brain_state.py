"""Check Exit Brain V3 state for TP levels."""
import sys
sys.path.insert(0, '/app')
import asyncio

# Load executor singleton
from backend.domains.exits.exit_brain_v3 import dynamic_executor_instance

if dynamic_executor_instance is None:
    print("❌ Exit Brain executor not initialized!")
    sys.exit(1)

state = dynamic_executor_instance._state

print("\n" + "="*80)
print("EXIT BRAIN V3 - STATE INSPECTION")
print("="*80)

if not state.positions:
    print("\n❌ No positions in state!")
else:
    print(f"\n✅ Found {len(state.positions)} positions in state\n")
    
    for symbol, pos_state in state.positions.items():
        print(f"{'─'*80}")
        print(f"💰 {symbol} {pos_state.side}")
        print(f"{'─'*80}")
        print(f"Entry:        ${pos_state.entry_price:.4f}")
        print(f"Size:         {pos_state.size}")
        print(f"Active SL:    ${pos_state.active_sl:.4f}" if pos_state.active_sl else "Active SL:    None")
        
        if pos_state.tp_legs:
            print(f"\n✅ TP Legs: {len(pos_state.tp_legs)}")
            for i, leg in enumerate(pos_state.tp_legs):
                triggered = "✅ TRIGGERED" if i in pos_state.triggered_legs else "⏳ Pending"
                print(f"  TP{i}: ${leg.price:.4f} ({leg.size_pct*100:.1f}%) - {triggered}")
        else:
            print(f"\n❌ NO TP LEGS!")
        
        if pos_state.tp_levels:
            print(f"\n✅ TP Levels (tuple format): {len(pos_state.tp_levels)}")
            for i, (price, size_pct) in enumerate(pos_state.tp_levels):
                print(f"  TP{i}: ${price:.4f} ({size_pct*100:.1f}%)")
        else:
            print(f"\n❌ NO TP LEVELS!")
        
        print(f"\nTriggered legs: {pos_state.triggered_legs}")
        print(f"Closed size: {pos_state.closed_size}")
        print()

print("="*80)

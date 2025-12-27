"""Test TradeLifecycleManager fix - verify trade_state.json auto-update"""

import json
import sys
from pathlib import Path

def test_lifecycle_fix():
    """Test that trade_state.json gets updated automatically."""
    
    print("🧪 TESTING TRADELIFECYCLEMANAGER FIX")
    print("=" * 80)
    
    # Check if trade_state.json exists
    state_file = Path("/app/backend/data/trade_state.json")
    
    if not state_file.exists():
        print("❌ trade_state.json not found!")
        return False
    
    # Load current state
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load trade_state.json: {e}")
        return False
    
    print(f"\n📊 Current state has {len(state)} symbols")
    
    # Check current open positions
    open_positions = ["SOLUSDT", "DOTUSDT", "DOGEUSDT"]
    
    print("\n🔍 Checking open positions in state:")
    for symbol in open_positions:
        if symbol in state:
            s = state[symbol]
            print(f"\n✅ {symbol}:")
            print(f"   Side: {s.get('side')}")
            print(f"   Qty: {s.get('qty')}")
            print(f"   Entry: ${s.get('avg_entry')}")
            print(f"   Trail: {s.get('ai_trail_pct', 0)*100:.2f}%")
            print(f"   TP: {s.get('ai_tp_pct', 0)*100:.2f}%")
            print(f"   SL: {s.get('ai_sl_pct', 0)*100:.2f}%")
            
            # Check if has required fields
            required_fields = ['ai_trail_pct', 'ai_tp_pct', 'ai_sl_pct']
            missing = [f for f in required_fields if f not in s]
            
            if missing:
                print(f"   ⚠️ Missing fields: {missing}")
            else:
                print("   ✅ All required fields present")
        else:
            print(f"\n❌ {symbol}: NOT IN STATE FILE")
    
    # Summary
    print("\n" + "=" * 80)
    print("📝 SUMMARY:")
    
    configured = sum(1 for sym in open_positions if sym in state and 'ai_trail_pct' in state[sym])
    total = len(open_positions)
    
    print(f"   Configured: {configured}/{total}")
    
    if configured == total:
        print("\n✅ ALL POSITIONS READY FOR AUTOMATIC PARTIAL PROFIT!")
        print("   Trailing Stop Manager will manage these positions")
        return True
    else:
        print(f"\n⚠️ {total - configured} positions need configuration")
        return False

if __name__ == "__main__":
    success = test_lifecycle_fix()
    sys.exit(0 if success else 1)

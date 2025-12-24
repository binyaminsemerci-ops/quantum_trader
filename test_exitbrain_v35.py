#!/usr/bin/env python3
"""
Direct test of ExitBrain v3.5 adaptive leverage computation
"""
import sys
import os

# Set up paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/microservices')
sys.path.insert(0, '/app/ai_engine')
os.environ['PYTHONPATH'] = '/app/backend:/app/microservices:/app/ai_engine:/app'

from domains.exits.exit_brain_v3.v35_integration import ExitBrainV35Integration
import json

def test_exitbrain_v35():
    """Test ExitBrain v3.5 with BTCUSDT trade"""
    print("=" * 80)
    print("🧪 TESTING ExitBrain v3.5 - Adaptive Leverage Engine")
    print("=" * 80)
    
    # Initialize ExitBrain v3.5
    exitbrain = ExitBrainV35Integration(enabled=True)
    print(f"✅ ExitBrainV35Integration initialized: enabled={exitbrain.enabled}")
    
    # Test payload - simulating a BTCUSDT trade
    test_payload = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "position_size_usd": 25.0,
        "leverage": 10,
        "entry_price": 95000.0,  # Example price
        "source": "v35_one_shot_test"
    }
    
    print(f"\n📊 Test Payload:")
    print(json.dumps(test_payload, indent=2))
    
    # Call compute_adaptive_levels
    print(f"\n🔬 Computing adaptive levels...")
    try:
        adaptive_levels = exitbrain.compute_adaptive_levels(
            symbol=test_payload["symbol"],
            side=test_payload["side"],
            position_size_usd=test_payload["position_size_usd"],
            base_leverage=test_payload["leverage"]
        )
        
        print(f"\n✅ ExitBrain v3.5 Output:")
        print(json.dumps(adaptive_levels, indent=2))
        
        # Check if target_leverage is NOT stuck at 1
        target_leverage = adaptive_levels.get("target_leverage", 1)
        print(f"\n🎯 Target Leverage: {target_leverage}")
        
        if target_leverage > 1:
            print(f"✅ SUCCESS: target_leverage = {target_leverage} (NOT stuck at 1)")
        else:
            print(f"⚠️  WARNING: target_leverage = {target_leverage} (might be stuck at 1)")
            
    except Exception as e:
        print(f"\n❌ ERROR in compute_adaptive_levels:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("🏁 Test Complete")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_exitbrain_v35()
    sys.exit(0 if success else 1)

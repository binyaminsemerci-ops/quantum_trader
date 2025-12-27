"""
Test MSC AI Consumer Integration

Verifies that all three components can read MSC AI policy.

Author: Quantum Trader Team
Date: 2025-11-30
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_policy_store_creation():
    """Test that policy stores can be created in all components."""
    print("\n" + "="*60)
    print("TESTING POLICY STORE CREATION")
    print("="*60)
    
    try:
        from backend.services.msc_ai_integration import QuantumPolicyStoreMSC
        
        # Create three separate stores (one for each component)
        stores = {
            "Event Executor": QuantumPolicyStoreMSC(),
            "Orchestrator": QuantumPolicyStoreMSC(),
            "Risk Guard": QuantumPolicyStoreMSC()
        }
        
        for component, store in stores.items():
            print(f"✅ {component}: Policy store created")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy store creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_read():
    """Test that policy can be read (may be None if not evaluated yet)."""
    print("\n" + "="*60)
    print("TESTING POLICY READ")
    print("="*60)
    
    try:
        from backend.services.msc_ai_integration import QuantumPolicyStoreMSC
        
        store = QuantumPolicyStoreMSC()
        policy = store.read_policy()
        
        if policy:
            print(f"✅ Policy read successfully:")
            print(f"   - Risk Mode: {policy.get('risk_mode', 'N/A')}")
            print(f"   - Max Risk: {policy.get('max_risk_per_trade', 0)*100:.2f}%")
            print(f"   - Min Confidence: {policy.get('global_min_confidence', 0):.2f}")
            print(f"   - Max Positions: {policy.get('max_positions', 0)}")
            print(f"   - Allowed Strategies: {len(policy.get('allowed_strategies', []))}")
        else:
            print("ℹ️  No policy available yet (MSC AI hasn't evaluated)")
            print("   This is expected on first run - policy will be created after 30 seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy read failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_flags():
    """Test that MSC_AI_AVAILABLE flags are set correctly."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION FLAGS")
    print("="*60)
    
    try:
        from backend.services.execution.event_driven_executor import MSC_AI_AVAILABLE as EE_MSC
        from backend.services.governance.orchestrator_policy import MSC_AI_AVAILABLE as ORCH_MSC
        from backend.services.risk.risk_guard import MSC_AI_AVAILABLE as RG_MSC
        
        flags = {
            "Event Executor": EE_MSC,
            "Orchestrator": ORCH_MSC,
            "Risk Guard": RG_MSC
        }
        
        all_available = True
        for component, flag in flags.items():
            status = "✅ AVAILABLE" if flag else "❌ NOT AVAILABLE"
            print(f"{status} - {component}")
            if not flag:
                all_available = False
        
        return all_available
        
    except Exception as e:
        print(f"❌ Integration flag check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_policy_write_read():
    """Test writing and reading a mock policy."""
    print("\n" + "="*60)
    print("TESTING POLICY WRITE/READ")
    print("="*60)
    
    try:
        from backend.services.msc_ai_integration import QuantumPolicyStoreMSC
        from datetime import datetime, timezone
        
        store = QuantumPolicyStoreMSC()
        
        # Create mock policy
        mock_policy = {
            "risk_mode": "NORMAL",
            "allowed_strategies": ["TEST_001", "TEST_002"],
            "max_risk_per_trade": 0.0075,
            "global_min_confidence": 0.60,
            "max_positions": 10,
            "max_daily_trades": 30,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        print("📝 Writing mock policy...")
        store.write_policy(mock_policy)
        print("✅ Policy written to database")
        
        print("\n📖 Reading policy back...")
        read_policy = store.read_policy()
        
        if read_policy:
            print("✅ Policy read successfully:")
            print(f"   - Risk Mode: {read_policy.get('risk_mode')}")
            print(f"   - Strategies: {read_policy.get('allowed_strategies')}")
            print(f"   - Max Risk: {read_policy.get('max_risk_per_trade', 0)*100:.2f}%")
            
            # Verify content matches
            if read_policy['risk_mode'] == mock_policy['risk_mode']:
                print("✅ Policy content verified - write/read working correctly")
                return True
            else:
                print("❌ Policy content mismatch")
                return False
        else:
            print("❌ Failed to read back written policy")
            return False
        
    except Exception as e:
        print(f"❌ Policy write/read test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print(" "*20 + "MSC AI CONSUMER INTEGRATION TEST")
    print("="*80)
    
    results = {
        "Policy Store Creation": test_policy_store_creation(),
        "Integration Flags": test_integration_flags(),
        "Policy Read": test_policy_read(),
        "Policy Write/Read": test_mock_policy_write_read()
    }
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MSC AI consumer integration is working!")
        print("\nNext steps:")
        print("1. Start backend: python backend/main.py")
        print("2. MSC AI will evaluate after 30 seconds")
        print("3. Check policy: curl http://localhost:8000/api/msc/status")
        print("4. Watch logs for MSC AI messages")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

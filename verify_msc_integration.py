"""
MSC AI Integration Verification Script

Run this script to verify that all MSC AI components are properly integrated.

Author: Quantum Trader Team
Date: 2025-11-30
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all MSC AI modules can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    tests = [
        ("Meta Strategy Controller Core", "backend.services.meta_strategy_controller"),
        ("MSC AI Integration", "backend.services.msc_ai_integration"),
        ("MSC AI Scheduler", "backend.services.msc_ai_scheduler"),
        ("MSC AI API Routes", "backend.routes.msc_ai"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_path in tests:
        try:
            __import__(module_path)
            print(f"✅ {name}: OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: ERROR - {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_controller_initialization():
    """Test MSC AI controller can be initialized."""
    print("\n" + "="*60)
    print("TESTING CONTROLLER INITIALIZATION")
    print("="*60)
    
    try:
        from backend.services.msc_ai_integration import get_msc_controller
        
        controller = get_msc_controller()
        
        print(f"✅ Controller initialized successfully")
        print(f"   - Evaluation period: {controller.evaluation_period_days} days")
        print(f"   - Min strategies: {controller.min_strategies}")
        print(f"   - Max strategies: {controller.max_strategies}")
        
        return True
        
    except Exception as e:
        print(f"❌ Controller initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_repositories():
    """Test that repositories can be created."""
    print("\n" + "="*60)
    print("TESTING REPOSITORIES")
    print("="*60)
    
    try:
        from backend.services.msc_ai_integration import (
            QuantumMetricsRepository,
            QuantumStrategyRepositoryMSC,
            QuantumPolicyStoreMSC
        )
        
        # Test MetricsRepository
        metrics_repo = QuantumMetricsRepository()
        print("✅ MetricsRepository created")
        
        # Test StrategyRepository
        strategy_repo = QuantumStrategyRepositoryMSC()
        print("✅ StrategyRepository created")
        
        # Test PolicyStore
        policy_store = QuantumPolicyStoreMSC()
        print("✅ PolicyStore created")
        
        return True
        
    except Exception as e:
        print(f"❌ Repository creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_routes():
    """Test that API routes are defined."""
    print("\n" + "="*60)
    print("TESTING API ROUTES")
    print("="*60)
    
    try:
        from backend.routes.msc_ai import router
        
        routes = [route.path for route in router.routes]
        
        print(f"✅ Router loaded with {len(routes)} routes:")
        for route in routes:
            print(f"   - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ API routes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler():
    """Test that scheduler can be created."""
    print("\n" + "="*60)
    print("TESTING SCHEDULER")
    print("="*60)
    
    try:
        from backend.services.msc_ai_scheduler import get_msc_scheduler
        
        scheduler = get_msc_scheduler()
        status = scheduler.get_status()
        
        print(f"✅ Scheduler created")
        print(f"   - Enabled: {status['enabled']}")
        print(f"   - Status: {status['status']}")
        
        if status['enabled']:
            print(f"   - Interval: {status.get('interval_minutes')} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_models():
    """Test that data models work correctly."""
    print("\n" + "="*60)
    print("TESTING DATA MODELS")
    print("="*60)
    
    try:
        from backend.services.meta_strategy_controller import (
            SystemHealth,
            GlobalPolicy,
            RiskMode
        )
        from datetime import datetime, timezone
        
        # Test SystemHealth (dataclass with positional args)
        health = SystemHealth(
            current_drawdown_pct=2.5,
            global_winrate=0.58,
            equity_curve_slope=0.8,
            global_regime="BULL_TRENDING",
            volatility_level="NORMAL",
            recent_trade_count=50,
            consecutive_losses=1,
            days_since_profit=0
        )
        print(f"✅ SystemHealth model works")
        print(f"   - Drawdown: {health.current_drawdown_pct}%")
        print(f"   - Winrate: {health.global_winrate * 100}%")
        
        # Test GlobalPolicy (dataclass)
        policy = GlobalPolicy(
            risk_mode=RiskMode.NORMAL,
            allowed_strategies=["TREND_001", "MOMENTUM_002"],
            max_risk_per_trade=0.0075,
            global_min_confidence=0.60,
            max_positions=10,
            max_daily_trades=30
        )
        print(f"✅ GlobalPolicy model works")
        print(f"   - Risk mode: {policy.risk_mode}")
        print(f"   - Max positions: {policy.max_positions}")
        
        # Test RiskMode enum
        assert RiskMode.DEFENSIVE == "DEFENSIVE"
        assert RiskMode.NORMAL == "NORMAL"
        assert RiskMode.AGGRESSIVE == "AGGRESSIVE"
        print(f"✅ RiskMode enum works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print(" "*20 + "MSC AI INTEGRATION VERIFICATION")
    print("="*80)
    
    results = {
        "Imports": test_imports(),
        "Controller": test_controller_initialization(),
        "Repositories": test_repositories(),
        "API Routes": test_api_routes(),
        "Scheduler": test_scheduler(),
        "Data Models": test_data_models()
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
        print("\n🎉 ALL TESTS PASSED! MSC AI is fully integrated and ready to use!")
        print("\nNext steps:")
        print("1. Start the backend: python backend/main.py")
        print("2. Check status: curl http://localhost:8000/api/msc/status")
        print("3. View documentation: MSC_AI_QUICKSTART.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

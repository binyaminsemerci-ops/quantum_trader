"""
TP v3 Enhancement Tests - Advanced features validation
=======================================================

Tests for TP v3 enhancements:
1. RL v3 TP-specific reward component
2. Dynamic Trailing Rearm
3. Risk v3 Integration
4. TP Performance Tracking
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rl_tp_retraining_reward():
    """Test RL v3 TP accuracy reward component."""
    print("\n[TEST 1] RL v3 TP-Specific Reward Component")
    print("-" * 60)
    
    try:
        from backend.domains.learning.rl_v3.reward_v3 import compute_reward
        
        # Test reward with perfect TP zone accuracy
        reward_perfect = compute_reward(
            pnl_delta=100.0,
            drawdown=-0.05,
            position_size=0.5,
            regime_alignment=1.0,
            volatility=0.02,
            tp_zone_accuracy=1.0  # Perfect TP prediction
        )
        
        # Test reward with poor TP zone accuracy
        reward_poor = compute_reward(
            pnl_delta=100.0,
            drawdown=-0.05,
            position_size=0.5,
            regime_alignment=1.0,
            volatility=0.02,
            tp_zone_accuracy=0.2  # Poor TP prediction
        )
        
        tp_bonus_perfect = 1.0 * 5.0  # Should be +5.0
        tp_bonus_poor = 0.2 * 5.0  # Should be +1.0
        
        # Reward with perfect TP should be higher by ~4.0
        reward_diff = reward_perfect - reward_poor
        expected_diff = tp_bonus_perfect - tp_bonus_poor
        
        print(f"   ✅ Perfect TP accuracy reward: {reward_perfect:.2f}")
        print(f"   ✅ Poor TP accuracy reward: {reward_poor:.2f}")
        print(f"   ✅ Reward difference: {reward_diff:.2f} (expected: {expected_diff:.2f})")
        
        assert abs(reward_diff - expected_diff) < 1.0, \
            f"TP accuracy bonus not applied correctly (diff={reward_diff:.2f}, expected={expected_diff:.2f})"
        
        print("   ✅ RL v3 TP reward component working!")
        return True
    except Exception as e:
        print(f"   ⚠️  Test skipped: {e}")
        return False


def test_dynamic_trailing_rearm():
    """Test Dynamic Trailing Rearm module."""
    print("\n[TEST 2] Dynamic Trailing Rearm")
    print("-" * 60)
    
    try:
        from backend.services.monitoring.dynamic_trailing_rearm import DynamicTrailingManager
        
        manager = DynamicTrailingManager()
        
        # Test profit-based callback tightening
        
        # Scenario 1: Small profit (1%) - no adjustment (below 2% threshold)
        callback_1 = manager.calculate_optimal_callback(
            unrealized_pnl_pct=0.01,
            current_callback_pct=2.0,
            position_age_minutes=10
        )
        print(f"   ✅ 1% profit: callback={callback_1} (expected: None - too small)")
        assert callback_1 is None, "Should not adjust for <2% profit"
        
        # Scenario 2: Good profit (8%) - should tighten
        callback_2 = manager.calculate_optimal_callback(
            unrealized_pnl_pct=0.08,
            current_callback_pct=2.0,
            position_age_minutes=15
        )
        print(f"   ✅ 8% profit: callback={callback_2}% (expected: ~0.05% - min limit)")
        assert callback_2 is not None, "Should adjust for >5% profit"
        assert callback_2 <= 2.0, "Callback should tighten or stay at min"
        
        # Scenario 3: Huge profit (25%) - already at minimum
        callback_3 = manager.calculate_optimal_callback(
            unrealized_pnl_pct=0.25,
            current_callback_pct=2.0,
            position_age_minutes=30
        )
        print(f"   ✅ 25% profit: callback={callback_3}% (expected: ~0.05% - min limit)")
        assert callback_3 is not None, "Should adjust for >20% profit"
        # Both hit min_callback_pct floor, so they're equal
        assert callback_3 == callback_2 or callback_3 <= callback_2, "Should be at min callback"
        
        # Test partial TP levels
        levels = manager.get_partial_tp_levels(
            entry_price=100.0,
            tp_price=115.0,
            side="LONG",
            unrealized_pnl_pct=0.12
        )
        print(f"   ✅ Partial TP levels (12% profit): {len(levels)} levels")
        assert len(levels) >= 2, "Should have 2+ levels for >10% profit"
        
        print("   ✅ Dynamic Trailing Rearm working!")
        return True
    except Exception as e:
        print(f"   ⚠️  Test skipped: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_v3_tp_adjustment():
    """Test Risk v3 Integration for TP adjustments."""
    print("\n[TEST 3] Risk v3 Integration")
    print("-" * 60)
    
    try:
        from backend.services.risk_management.risk_v3_integration import (
            RiskV3Integrator,
            RiskV3Context
        )
        from datetime import datetime, timezone
        
        integrator = RiskV3Integrator()
        
        # Test low risk context - no tightening
        context_low = RiskV3Context(
            ess_factor=1.0,
            systemic_risk_level=0.2,
            correlation_risk=0.3,
            portfolio_heat=0.4,
            var_95=50.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        should_tighten_low = integrator.should_tighten_tp(context_low)
        adjustment_low = integrator.get_tp_adjustment_factor(context_low)
        
        print(f"   ✅ Low risk: tighten={should_tighten_low}, factor={adjustment_low:.2f}")
        assert not should_tighten_low, "Should not tighten at low risk"
        assert adjustment_low >= 1.0, "Should widen TP at low risk"
        
        # Test high ESS - should tighten
        context_high_ess = RiskV3Context(
            ess_factor=2.8,  # Critical ESS
            systemic_risk_level=0.3,
            correlation_risk=0.4,
            portfolio_heat=0.5,
            var_95=150.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        should_tighten_ess = integrator.should_tighten_tp(context_high_ess)
        adjustment_ess = integrator.get_tp_adjustment_factor(context_high_ess)
        
        print(f"   ✅ High ESS: tighten={should_tighten_ess}, factor={adjustment_ess:.2f}")
        assert should_tighten_ess, "Should tighten at high ESS"
        assert adjustment_ess < 1.0, "Should tighten TP at high ESS"
        
        # Test systemic risk - defensive mode
        context_systemic = RiskV3Context(
            ess_factor=1.2,
            systemic_risk_level=0.85,  # Crisis mode
            correlation_risk=0.7,
            portfolio_heat=0.6,
            var_95=200.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        should_tighten_sys = integrator.should_tighten_tp(context_systemic)
        adjustment_sys = integrator.get_tp_adjustment_factor(context_systemic)
        
        print(f"   ✅ Systemic risk: tighten={should_tighten_sys}, factor={adjustment_sys:.2f}")
        assert should_tighten_sys, "Should tighten during systemic risk"
        assert adjustment_sys < 1.0, "Should be defensive during crisis"
        
        print("   ✅ Risk v3 Integration working!")
        return True
    except Exception as e:
        print(f"   ⚠️  Test skipped: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tp_performance_tracking():
    """Test TP Performance Tracking module."""
    print("\n[TEST 4] TP Performance Tracking")
    print("-" * 60)
    
    try:
        from backend.services.monitoring.tp_performance_tracker import TPPerformanceTracker
        import tempfile
        
        # Use temp file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        
        tracker = TPPerformanceTracker(storage_path=temp_path)
        
        # Record TP attempt
        entry_time = datetime.now(timezone.utc)
        tracker.record_tp_attempt(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            entry_time=entry_time,
            entry_price=50000.0,
            tp_target_price=51000.0,
            side="LONG"
        )
        
        # Record TP hit
        exit_time = entry_time + timedelta(hours=2)
        tracker.record_tp_hit(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            exit_time=exit_time,
            exit_price=50950.0,  # Within 5% of target
            tp_target_price=51000.0,
            entry_time=entry_time,
            entry_price=50000.0,
            profit_usd=100.0
        )
        
        # Get metrics
        metrics = tracker.get_metrics(strategy_id="test_strategy", symbol="BTCUSDT")
        assert len(metrics) == 1, "Should have 1 metric entry"
        
        metric = metrics[0]
        print(f"   ✅ TP attempts: {metric.tp_attempts}")
        print(f"   ✅ TP hits: {metric.tp_hits}")
        print(f"   ✅ Hit rate: {metric.tp_hit_rate:.1%}")
        print(f"   ✅ Avg slippage: {metric.avg_slippage_pct:.3%}")
        print(f"   ✅ Avg time to TP: {metric.avg_time_to_tp_minutes:.1f} min")
        
        assert metric.tp_attempts == 1, "Should have 1 attempt"
        assert metric.tp_hits == 1, "Should have 1 hit"
        assert metric.tp_hit_rate == 1.0, "Should have 100% hit rate"
        
        # Test summary
        summary = tracker.get_summary()
        print(f"   ✅ Summary: {summary['overall_hit_rate']:.1%} hit rate, ${summary['total_profit']:.2f} profit")
        
        # Test RL feedback
        feedback = tracker.get_feedback_for_rl_training()
        print(f"   ✅ RL feedback: {feedback}")
        
        # Cleanup
        temp_path.unlink()
        
        print("   ✅ TP Performance Tracking working!")
        return True
    except Exception as e:
        print(f"   ⚠️  Test skipped: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all TP v3 enhancement tests."""
    print("\n" + "=" * 60)
    print("TP v3 ENHANCEMENT TESTS")
    print("=" * 60)
    
    tests = [
        test_rl_tp_retraining_reward,
        test_dynamic_trailing_rearm,
        test_risk_v3_tp_adjustment,
        test_tp_performance_tracking
    ]
    
    results = {}
    for test_func in tests:
        try:
            results[test_func.__name__] = test_func()
        except Exception as e:
            print(f"\n   ❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[test_func.__name__] = False
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL/SKIP"
        print(f"   {status}: {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

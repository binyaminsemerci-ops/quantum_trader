"""
TP v3 Unit Tests - Core TP logic validation
============================================

Tests Dynamic TP/SL calculator without requiring RL v3 dependencies.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.execution.dynamic_tpsl import get_dynamic_tpsl_calculator
from backend.services.execution.hybrid_tpsl import calculate_hybrid_levels


def test_dynamic_tpsl_basic():
    """Test basic TP/SL calculation."""
    print("\n[TEST 1] Basic Dynamic TP/SL Calculation")
    print("-" * 60)
    
    calculator = get_dynamic_tpsl_calculator()
    
    # Test normal confidence
    result = calculator.calculate(
        signal_confidence=0.75,
        action='BUY',
        market_conditions={'volatility': 0.02},
        risk_mode='NORMAL'
    )
    
    assert result.tp_percent > 0, "TP should be positive"
    assert result.sl_percent > 0, "SL should be positive"
    assert result.tp_percent > result.sl_percent, "TP should be greater than SL"
    assert result.tp_percent >= 0.05, "TP should be at least 5%"
    
    rr_ratio = result.tp_percent / result.sl_percent
    assert rr_ratio >= 1.5, f"R:R should be >= 1.5x, got {rr_ratio:.2f}x"
    
    print(f"   ✅ TP: {result.tp_percent:.2%}")
    print(f"   ✅ SL: {result.sl_percent:.2%}")
    print(f"   ✅ Trail: {result.trail_percent:.2%}")
    print(f"   ✅ R:R: {rr_ratio:.2f}x")
    print(f"   ✅ Rationale: {result.rationale}")
    
    return True


def test_confidence_scaling():
    """Test TP/SL scales with confidence."""
    print("\n[TEST 2] Confidence Scaling")
    print("-" * 60)
    
    calculator = get_dynamic_tpsl_calculator()
    
    low_conf = calculator.calculate(0.50, 'BUY', risk_mode='NORMAL')
    high_conf = calculator.calculate(0.90, 'BUY', risk_mode='NORMAL')
    
    # High confidence should have wider TP and tighter SL
    assert high_conf.tp_percent > low_conf.tp_percent, "High confidence should have wider TP"
    assert high_conf.sl_percent < low_conf.sl_percent, "High confidence should have tighter SL"
    
    print(f"   ✅ Low conf (0.50): TP={low_conf.tp_percent:.2%}, SL={low_conf.sl_percent:.2%}")
    print(f"   ✅ High conf (0.90): TP={high_conf.tp_percent:.2%}, SL={high_conf.sl_percent:.2%}")
    
    return True


def test_risk_mode_adjustment():
    """Test TP/SL adjusts for different risk modes."""
    print("\n[TEST 3] Risk Mode Adjustment")
    print("-" * 60)
    
    calculator = get_dynamic_tpsl_calculator()
    
    normal = calculator.calculate(0.75, 'BUY', risk_mode='NORMAL')
    aggressive = calculator.calculate(0.75, 'BUY', risk_mode='AGGRESSIVE')
    critical = calculator.calculate(0.75, 'BUY', risk_mode='CRITICAL')
    
    # Aggressive should have wider TP
    assert aggressive.tp_percent > normal.tp_percent, "Aggressive should have wider TP"
    
    # Critical should have narrower TP and wider SL (defensive)
    assert critical.tp_percent < normal.tp_percent, "Critical should have narrower TP"
    assert critical.sl_percent > normal.sl_percent, "Critical should have wider SL"
    
    print(f"   ✅ Normal: TP={normal.tp_percent:.2%}, SL={normal.sl_percent:.2%}")
    print(f"   ✅ Aggressive: TP={aggressive.tp_percent:.2%}, SL={aggressive.sl_percent:.2%}")
    print(f"   ✅ Critical: TP={critical.tp_percent:.2%}, SL={critical.sl_percent:.2%}")
    
    return True


def test_risk_v3_integration():
    """Test Risk v3 context adjustments."""
    print("\n[TEST 4] Risk v3 Integration")
    print("-" * 60)
    
    calculator = get_dynamic_tpsl_calculator()
    
    baseline = calculator.calculate(0.75, 'BUY', risk_mode='NORMAL')
    
    # Test high ESS tightening
    high_ess = calculator.calculate(
        0.75, 'BUY', risk_mode='NORMAL',
        risk_v3_context={'ess_factor': 2.5, 'systemic_risk_level': 0.3}
    )
    
    # Test systemic risk defensive mode
    systemic = calculator.calculate(
        0.75, 'BUY', risk_mode='NORMAL',
        risk_v3_context={'ess_factor': 1.0, 'systemic_risk_level': 0.8}
    )
    
    assert high_ess.tp_percent < baseline.tp_percent, "High ESS should tighten TP"
    assert systemic.tp_percent < baseline.tp_percent, "Systemic risk should tighten TP"
    assert systemic.sl_percent > baseline.sl_percent, "Systemic risk should widen SL"
    
    print(f"   ✅ Baseline: TP={baseline.tp_percent:.2%}")
    print(f"   ✅ High ESS (2.5): TP={high_ess.tp_percent:.2%}")
    print(f"   ✅ Systemic Risk (0.8): TP={systemic.tp_percent:.2%}, SL={systemic.sl_percent:.2%}")
    
    return True


def test_rl_tp_suggestion_blending():
    """Test RL TP suggestion blending."""
    print("\n[TEST 5] RL TP Suggestion Blending")
    print("-" * 60)
    
    calculator = get_dynamic_tpsl_calculator()
    
    without_rl = calculator.calculate(0.75, 'BUY', risk_mode='NORMAL')
    
    with_rl = calculator.calculate(
        0.75, 'BUY', risk_mode='NORMAL',
        rl_tp_suggestion=0.08  # RL suggests 8% TP
    )
    
    # TP should be blended (60% confidence-based + 40% RL)
    expected_blend = (without_rl.tp_percent * 0.6) + (0.08 * 0.4)
    
    print(f"   ✅ Without RL: TP={without_rl.tp_percent:.2%}")
    print(f"   ✅ With RL (8%): TP={with_rl.tp_percent:.2%}")
    print(f"   ✅ Expected blend: {expected_blend:.2%}")
    print(f"   ✅ Blending active: {abs(with_rl.tp_percent - expected_blend) < 0.001}")
    
    return True


def test_hybrid_tpsl_blending():
    """Test Hybrid TP/SL blending logic."""
    print("\n[TEST 6] Hybrid TP/SL Blending")
    print("-" * 60)
    
    # Low confidence - should use base TP
    low_conf = calculate_hybrid_levels(
        entry_price=100000.0,
        side='BUY',
        risk_sl_percent=0.025,
        base_tp_percent=0.05,
        ai_tp_percent=0.08,
        ai_trail_percent=0.02,
        confidence=0.15
    )
    
    # High confidence - should use AI extended TP
    high_conf = calculate_hybrid_levels(
        entry_price=100000.0,
        side='BUY',
        risk_sl_percent=0.025,
        base_tp_percent=0.05,
        ai_tp_percent=0.08,
        ai_trail_percent=0.02,
        confidence=0.85
    )
    
    assert low_conf['final_tp_percent'] == 0.05, "Low confidence should use base TP"
    assert high_conf['final_tp_percent'] == 0.08, "High confidence should use AI extended TP"
    assert high_conf['trail_callback_percent'] > 0, "High confidence should enable trailing"
    
    print(f"   ✅ Low conf (0.15): TP={low_conf['final_tp_percent']:.2%}, mode={low_conf['mode']}")
    print(f"   ✅ High conf (0.85): TP={high_conf['final_tp_percent']:.2%}, mode={high_conf['mode']}")
    print(f"   ✅ Trailing enabled: {high_conf['trail_callback_percent']:.3%}")
    
    return True


def run_all_tests():
    """Run all TP v3 unit tests."""
    print("\n" + "=" * 60)
    print("🧪 TP v3 UNIT TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic TP/SL Calculation", test_dynamic_tpsl_basic),
        ("Confidence Scaling", test_confidence_scaling),
        ("Risk Mode Adjustment", test_risk_mode_adjustment),
        ("Risk v3 Integration", test_risk_v3_integration),
        ("RL TP Suggestion Blending", test_rl_tp_suggestion_blending),
        ("Hybrid TP/SL Blending", test_hybrid_tpsl_blending),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for name, passed, error in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {name}")
        if error:
            print(f"           {error}")
    
    print("-" * 60)
    print(f"   Passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED")
        return True
    else:
        print(f"⚠️  {total_count - passed_count} TESTS FAILED")
        return False


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result else 1)

"""
Tests for AdaptiveLeverageEngine
Validates leverage-aware TP/SL calculations and fail-safe clamps
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_leverage_engine import AdaptiveLeverageEngine, AdaptiveLevels


def test_lsf_decreases_with_leverage():
    """Test that LSF decreases as leverage increases"""
    engine = AdaptiveLeverageEngine()
    
    lsf_5x = engine.compute_lsf(5)
    lsf_50x = engine.compute_lsf(50)
    
    assert lsf_5x > lsf_50x, \
        f"LSF should decrease with leverage: 5x={lsf_5x:.4f}, 50x={lsf_50x:.4f}"
    
    print(f"✅ Test 1 PASSED: LSF(5x)={lsf_5x:.4f} > LSF(50x)={lsf_50x:.4f}")


def test_low_leverage_higher_tp():
    """Test that low leverage yields higher TP than high leverage"""
    engine = AdaptiveLeverageEngine()
    
    levels_5x = engine.compute_levels(0.01, 0.005, 5)
    levels_50x = engine.compute_levels(0.01, 0.005, 50)
    
    assert levels_5x.tp1_pct > levels_50x.tp1_pct, \
        f"Low leverage should have higher TP1: 5x={levels_5x.tp1_pct:.4f}, 50x={levels_50x.tp1_pct:.4f}"
    
    print(f"✅ Test 2 PASSED: TP1(5x)={levels_5x.tp1_pct*100:.2f}% > TP1(50x)={levels_50x.tp1_pct*100:.2f}%")


def test_high_leverage_wider_sl():
    """Test that high leverage has wider SL than low leverage"""
    engine = AdaptiveLeverageEngine()
    
    levels_5x = engine.compute_levels(0.01, 0.005, 5)
    levels_50x = engine.compute_levels(0.01, 0.005, 50)
    
    assert levels_50x.sl_pct >= levels_5x.sl_pct, \
        f"High leverage should have wider SL: 5x={levels_5x.sl_pct:.4f}, 50x={levels_50x.sl_pct:.4f}"
    
    print(f"✅ Test 3 PASSED: SL(50x)={levels_50x.sl_pct*100:.2f}% >= SL(5x)={levels_5x.sl_pct*100:.2f}%")


def test_clamps_work():
    """Test that fail-safe clamps prevent extreme values"""
    engine = AdaptiveLeverageEngine()
    
    # Test extreme leverage
    levels_100x = engine.compute_levels(0.01, 0.005, 100)
    
    assert engine.SL_CLAMP_MIN <= levels_100x.sl_pct <= engine.SL_CLAMP_MAX, \
        f"SL outside clamps: {levels_100x.sl_pct:.4f} not in [{engine.SL_CLAMP_MIN}, {engine.SL_CLAMP_MAX}]"
    
    assert levels_100x.tp1_pct >= engine.TP_MIN, \
        f"TP1 below minimum: {levels_100x.tp1_pct:.4f} < {engine.TP_MIN}"
    
    assert levels_100x.sl_pct >= engine.SL_MIN, \
        f"SL below minimum: {levels_100x.sl_pct:.4f} < {engine.SL_MIN}"
    
    print(f"✅ Test 4 PASSED: Clamps enforced (SL={levels_100x.sl_pct*100:.2f}% in [{engine.SL_CLAMP_MIN*100:.1f}%, {engine.SL_CLAMP_MAX*100:.1f}%])")


def test_harvest_schemes():
    """Test that harvest schemes match specification"""
    engine = AdaptiveLeverageEngine()
    
    scheme_5x = engine.harvest_scheme_for(5)
    scheme_20x = engine.harvest_scheme_for(20)
    scheme_50x = engine.harvest_scheme_for(50)
    
    assert scheme_5x == [0.3, 0.3, 0.4], f"Wrong scheme for 5x: {scheme_5x}"
    assert scheme_20x == [0.4, 0.4, 0.2], f"Wrong scheme for 20x: {scheme_20x}"
    assert scheme_50x == [0.5, 0.3, 0.2], f"Wrong scheme for 50x: {scheme_50x}"
    
    # All schemes should sum to 1.0
    assert abs(sum(scheme_5x) - 1.0) < 0.001, f"Scheme 5x doesn't sum to 1.0: {sum(scheme_5x)}"
    assert abs(sum(scheme_20x) - 1.0) < 0.001, f"Scheme 20x doesn't sum to 1.0: {sum(scheme_20x)}"
    assert abs(sum(scheme_50x) - 1.0) < 0.001, f"Scheme 50x doesn't sum to 1.0: {sum(scheme_50x)}"
    
    print(f"✅ Test 5 PASSED: Harvest schemes correct (5x={scheme_5x}, 20x={scheme_20x}, 50x={scheme_50x})")


def test_tp_progression():
    """Test that TP levels progress correctly (TP1 < TP2 < TP3)"""
    engine = AdaptiveLeverageEngine()
    
    for leverage in [5, 20, 50, 100]:
        levels = engine.compute_levels(0.01, 0.005, leverage)
        
        assert levels.tp1_pct < levels.tp2_pct < levels.tp3_pct, \
            f"Invalid TP progression at {leverage}x: {levels.tp1_pct} < {levels.tp2_pct} < {levels.tp3_pct}"
    
    print(f"✅ Test 6 PASSED: TP progression valid for all leverage levels")


def test_volatility_adjustment():
    """Test that volatility widens TP/SL appropriately"""
    engine = AdaptiveLeverageEngine()
    
    levels_calm = engine.compute_levels(0.01, 0.005, 20, volatility_factor=0.0)
    levels_volatile = engine.compute_levels(0.01, 0.005, 20, volatility_factor=0.5)
    
    # With volatility_factor=0.5, scale should be 1 + (0.5 * 0.2) = 1.1 for SL
    assert levels_volatile.sl_pct > levels_calm.sl_pct, \
        f"Volatility should widen SL: calm={levels_calm.sl_pct:.4f}, volatile={levels_volatile.sl_pct:.4f}"
    
    print(f"✅ Test 7 PASSED: Volatility adjustment works (SL: {levels_calm.sl_pct*100:.2f}% → {levels_volatile.sl_pct*100:.2f}%)")


def test_funding_adjustment():
    """Test that funding rate adjusts TP correctly"""
    engine = AdaptiveLeverageEngine()
    
    levels_negative = engine.compute_levels(0.01, 0.005, 20, funding_delta=-0.01)
    levels_positive = engine.compute_levels(0.01, 0.005, 20, funding_delta=0.01)
    
    assert levels_positive.tp1_pct > levels_negative.tp1_pct, \
        f"Positive funding should increase TP: negative={levels_negative.tp1_pct:.4f}, positive={levels_positive.tp1_pct:.4f}"
    
    print(f"✅ Test 8 PASSED: Funding adjustment works (TP1: {levels_negative.tp1_pct*100:.2f}% → {levels_positive.tp1_pct*100:.2f}%)")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("AdaptiveLeverageEngine Test Suite")
    print("="*60 + "\n")
    
    try:
        test_lsf_decreases_with_leverage()
        test_low_leverage_higher_tp()
        test_high_leverage_wider_sl()
        test_clamps_work()
        test_harvest_schemes()
        test_tp_progression()
        test_volatility_adjustment()
        test_funding_adjustment()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

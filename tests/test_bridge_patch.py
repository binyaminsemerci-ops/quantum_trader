#!/usr/bin/env python3
"""
BRIDGE-PATCH Smoke Test - Verify AI Sizer + Risk Governor integration

Usage:
    python tests/test_bridge_patch.py
    pytest tests/test_bridge_patch.py -v
"""

import sys
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from microservices.ai_engine.ai_sizer_policy import AISizerPolicy, SizingConfig, HarvestMode
from services.risk_governor import RiskGovernor, GovernorConfig


def test_ai_sizer_shadow_mode():
    """Test AI sizer in shadow mode (no execution)"""
    config = SizingConfig(ai_sizing_mode="shadow")
    sizer = AISizerPolicy(config)
    
    # Compute sizing for high-confidence signal
    leverage, size_usd, policy = sizer.compute_size_and_leverage(
        signal_confidence=0.95,
        volatility_factor=1.0,
        account_equity=10000.0
    )
    
    assert 5 <= leverage <= 80, f"Leverage {leverage} out of bounds [5..80]"
    assert size_usd > 0, "Size must be positive"
    assert policy['mode'] in [m.value for m in HarvestMode], "Invalid policy mode"
    
    print("✅ test_ai_sizer_shadow_mode PASSED")
    print(f"   Sizing: ${size_usd:.0f} @ {leverage:.1f}x, policy={policy['mode']}")


def test_ai_sizer_confidence_based_sizing():
    """Test that sizing scales with confidence"""
    config = SizingConfig(ai_sizing_mode="shadow")
    sizer = AISizerPolicy(config)
    
    # Low confidence
    lev_low, size_low, policy_low = sizer.compute_size_and_leverage(0.5, 1.0, 10000)
    
    # High confidence
    lev_high, size_high, policy_high = sizer.compute_size_and_leverage(0.95, 1.0, 10000)
    
    # High confidence should give higher leverage and typically larger size
    assert lev_high > lev_low, "High confidence should have higher leverage"
    assert policy_low['mode'] == 'scalper', "Low confidence should be scalper mode"
    assert policy_high['mode'] == 'trend_runner', "High confidence should be trend_runner"
    
    print("✅ test_ai_sizer_confidence_based_sizing PASSED")
    print(f"   Low conf: {lev_low:.1f}x ({policy_low['mode']}), High conf: {lev_high:.1f}x ({policy_high['mode']})")


def test_risk_governor_accept():
    """Test that governor accepts valid trades"""
    config = GovernorConfig(fail_open=False)
    gov = RiskGovernor(config)
    
    approved, reason, metadata = gov.evaluate(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.85,
        position_size_usd=500.0,
        leverage=10.0
    )
    
    assert approved, f"Governor should approve valid trade: {reason}"
    assert metadata['clamped_size_usd'] > 0, "Clamped size should be positive"
    
    print("✅ test_risk_governor_accept PASSED")
    print(f"   Reason: {reason}")


def test_risk_governor_clamps_leverage():
    """Test that governor clamps excessive leverage"""
    config = GovernorConfig(fail_open=False)
    gov = RiskGovernor(config)
    
    # Request excessive leverage (150x > max 80x)
    approved, reason, metadata = gov.evaluate(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.9,
        position_size_usd=1000.0,
        leverage=150.0  # Excessive
    )
    
    assert approved, "Should still approve (leverage gets clamped)"
    assert metadata['clamped_leverage'] <= 80, "Leverage should be clamped to max"
    assert metadata['clamped_leverage'] == 80, "Should clamp to exactly 80"
    
    print("✅ test_risk_governor_clamps_leverage PASSED")
    print(f"   Clamped: {metadata['clamped_leverage']:.1f}x (from 150x request)")


def test_risk_governor_clamps_size():
    """Test that governor clamps size to bounds"""
    config = GovernorConfig(min_order_usd=50, max_position_usd=10000)
    gov = RiskGovernor(config)
    
    # Request very large size (100k > max 10k)
    approved, reason, metadata = gov.evaluate(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.9,
        position_size_usd=100000.0  # Way over max
    )
    
    assert approved, "Should approve (size gets clamped)"
    assert metadata['clamped_size_usd'] <= 10000, "Size should be clamped to max"
    
    print("✅ test_risk_governor_clamps_size PASSED")
    print(f"   Clamped: ${metadata['clamped_size_usd']:.0f} (from $100k request)")


def test_risk_governor_rejects_notional_excess():
    """Test that governor rejects if notional exceeds limit"""
    config = GovernorConfig(max_notional_usd=50000)
    gov = RiskGovernor(config)
    
    # Request trade with notional 100k (size 10k @ 10x)
    approved, reason, metadata = gov.evaluate(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.9,
        position_size_usd=10000.0,
        leverage=10.0  # 100k notional > 50k limit
    )
    
    assert not approved, "Should reject excessive notional"
    assert "NOTIONAL_EXCEEDED" in reason, f"Wrong rejection reason: {reason}"
    
    print("✅ test_risk_governor_rejects_notional_excess PASSED")
    print(f"   Rejected: {reason}")


def test_risk_governor_confidence_floor():
    """Test that governor respects confidence floor"""
    config = GovernorConfig(min_confidence=0.75, fail_open=False)
    gov = RiskGovernor(config)
    
    # Signal below confidence floor
    approved, reason, metadata = gov.evaluate(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.5,  # Below 0.75 floor
        position_size_usd=500.0,
        leverage=5.0
    )
    
    assert not approved, "Should reject signal below confidence floor"
    assert "CONFIDENCE_BELOW_FLOOR" in reason, f"Wrong reason: {reason}"
    
    print("✅ test_risk_governor_confidence_floor PASSED")
    print(f"   Rejected: {reason}")


def test_end_to_end_shadow_flow():
    """Test full flow: AI sizer → payload injection → governor"""
    # AI sizer creates recommendation
    sizer_config = SizingConfig(ai_sizing_mode="shadow")
    sizer = AISizerPolicy(sizer_config)
    
    ai_lev, ai_size, ai_policy = sizer.compute_size_and_leverage(
        signal_confidence=0.88,
        volatility_factor=1.0,
        account_equity=10000.0
    )
    
    # Simulate payload after AI sizer injection
    trade_intent = {
        'symbol': 'ETHUSDT',
        'action': 'BUY',
        'confidence': 0.88,
        'ai_leverage': ai_lev,
        'ai_size_usd': ai_size,
        'ai_harvest_policy': ai_policy
    }
    
    # Governor evaluates
    gov = RiskGovernor(GovernorConfig())
    approved, reason, metadata = gov.evaluate(
        symbol=trade_intent['symbol'],
        action=trade_intent['action'],
        confidence=trade_intent['confidence'],
        position_size_usd=trade_intent['ai_size_usd'],
        leverage=trade_intent['ai_leverage']
    )
    
    assert approved, f"Should approve AI-sized trade: {reason}"
    assert metadata['clamped_leverage'] == ai_lev, "Leverage should match AI recommendation"
    
    print("✅ test_end_to_end_shadow_flow PASSED")
    print(f"   AI: ${ai_size:.0f} @ {ai_lev:.1f}x → Governor: ACCEPT")


def run_all_tests():
    """Run all smoke tests"""
    tests = [
        test_ai_sizer_shadow_mode,
        test_ai_sizer_confidence_based_sizing,
        test_risk_governor_accept,
        test_risk_governor_clamps_leverage,
        test_risk_governor_clamps_size,
        test_risk_governor_rejects_notional_excess,
        test_risk_governor_confidence_floor,
        test_end_to_end_shadow_flow,
    ]
    
    print("="*70)
    print("BRIDGE-PATCH SMOKE TESTS")
    print("="*70)
    print()
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        print("\n❌ Some tests failed")
        sys.exit(1)
    else:
        print("\n✅ All BRIDGE-PATCH tests PASSED - ready to deploy\n")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()

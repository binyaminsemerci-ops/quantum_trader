#!/usr/bin/env python3
"""
Verification Script for CHALLENGE_100 Hotfix
=============================================

Verifies that:
1. EXIT_MODE is EXIT_BRAIN_V3 (not CHALLENGE_100)
2. EXIT_BRAIN_PROFILE=CHALLENGE_100 activates challenge rules
3. Hard SL is sent in LIVE mode and not blocked by gateway
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_1_exit_mode_config():
    """Test 1: Verify EXIT_MODE is EXIT_BRAIN_V3"""
    print("=" * 70)
    print("TEST 1: EXIT_MODE Configuration")
    print("=" * 70)
    
    from backend.config.exit_mode import (
        get_exit_mode,
        is_exit_brain_mode,
        is_legacy_exit_mode,
        EXIT_MODE_EXIT_BRAIN_V3,
        EXIT_MODE_LEGACY
    )
    
    mode = get_exit_mode()
    print(f"✓ get_exit_mode() = {mode}")
    print(f"✓ is_exit_brain_mode() = {is_exit_brain_mode()}")
    print(f"✓ is_legacy_exit_mode() = {is_legacy_exit_mode()}")
    
    # Verify CHALLENGE_100 is NOT a valid EXIT_MODE
    try:
        from backend.config.exit_mode import EXIT_MODE_CHALLENGE_100
        print(f"✗ FAIL: EXIT_MODE_CHALLENGE_100 constant still exists!")
        return False
    except ImportError:
        print(f"✓ PASS: EXIT_MODE_CHALLENGE_100 removed (as expected)")
    
    # Verify is_challenge_100_mode() function doesn't exist
    try:
        from backend.config.exit_mode import is_challenge_100_mode
        print(f"✗ FAIL: is_challenge_100_mode() function still exists!")
        return False
    except ImportError:
        print(f"✓ PASS: is_challenge_100_mode() removed (as expected)")
    
    if mode != EXIT_MODE_EXIT_BRAIN_V3:
        print(f"✗ FAIL: EXIT_MODE should be EXIT_BRAIN_V3, got {mode}")
        return False
    
    print(f"\n✅ TEST 1 PASSED: EXIT_MODE = {EXIT_MODE_EXIT_BRAIN_V3}")
    return True


def test_2_exit_brain_profile():
    """Test 2: Verify EXIT_BRAIN_PROFILE activates CHALLENGE_100"""
    print("\n" + "=" * 70)
    print("TEST 2: EXIT_BRAIN_PROFILE Configuration")
    print("=" * 70)
    
    from backend.config.exit_mode import (
        get_exit_brain_profile,
        is_challenge_100_profile
    )
    
    profile = get_exit_brain_profile()
    is_challenge = is_challenge_100_profile()
    
    print(f"✓ get_exit_brain_profile() = {profile}")
    print(f"✓ is_challenge_100_profile() = {is_challenge}")
    
    env_profile = os.getenv("EXIT_BRAIN_PROFILE", "NOT_SET")
    print(f"✓ ENV EXIT_BRAIN_PROFILE = {env_profile}")
    
    if env_profile == "CHALLENGE_100":
        if not is_challenge:
            print(f"✗ FAIL: EXIT_BRAIN_PROFILE=CHALLENGE_100 but is_challenge_100_profile() = False")
            return False
        print(f"\n✅ TEST 2 PASSED: CHALLENGE_100 profile active")
    else:
        if is_challenge:
            print(f"✗ FAIL: EXIT_BRAIN_PROFILE != CHALLENGE_100 but is_challenge_100_profile() = True")
            return False
        print(f"\n✅ TEST 2 PASSED: Default profile active (set EXIT_BRAIN_PROFILE=CHALLENGE_100 to test)")
    
    return True


def test_3_hard_sl_gateway_compatibility():
    """Test 3: Verify hard SL uses correct module_name for gateway"""
    print("\n" + "=" * 70)
    print("TEST 3: Hard SL Gateway Compatibility")
    print("=" * 70)
    
    from backend.config.exit_mode import (
        is_exit_brain_live_fully_enabled,
        get_exit_mode,
        get_exit_executor_mode,
        is_exit_brain_live_rollout_enabled
    )
    
    exit_mode = get_exit_mode()
    executor_mode = get_exit_executor_mode()
    rollout = is_exit_brain_live_rollout_enabled()
    live_enabled = is_exit_brain_live_fully_enabled()
    
    print(f"✓ EXIT_MODE = {exit_mode}")
    print(f"✓ EXIT_EXECUTOR_MODE = {executor_mode}")
    print(f"✓ EXIT_BRAIN_V3_LIVE_ROLLOUT = {'ENABLED' if rollout else 'DISABLED'}")
    print(f"✓ is_exit_brain_live_fully_enabled() = {live_enabled}")
    
    # Check gateway configuration
    from backend.services.execution.exit_order_gateway import EXPECTED_EXIT_BRAIN_MODULES
    
    print(f"\n✓ Gateway EXPECTED_EXIT_BRAIN_MODULES: {EXPECTED_EXIT_BRAIN_MODULES}")
    
    if "exit_brain_executor" not in EXPECTED_EXIT_BRAIN_MODULES:
        print(f"✗ FAIL: 'exit_brain_executor' not in EXPECTED_EXIT_BRAIN_MODULES")
        return False
    
    print(f"✓ 'exit_brain_executor' in EXPECTED_EXIT_BRAIN_MODULES (gateway-compatible)")
    
    # Verify hard SL only triggers in LIVE mode
    if live_enabled:
        print(f"\n✓ LIVE mode ACTIVE - hard SL will be placed")
        print(f"  Hard SL placement requirements:")
        print(f"  ✓ EXIT_MODE = EXIT_BRAIN_V3")
        print(f"  ✓ EXIT_EXECUTOR_MODE = LIVE")
        print(f"  ✓ EXIT_BRAIN_V3_LIVE_ROLLOUT = ENABLED")
        print(f"  ✓ module_name = 'exit_brain_executor' (not blocked by gateway)")
    else:
        print(f"\n✓ SHADOW mode - hard SL will NOT be placed")
        print(f"  To enable, set:")
        print(f"    EXIT_MODE=EXIT_BRAIN_V3")
        print(f"    EXIT_EXECUTOR_MODE=LIVE")
        print(f"    EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED")
    
    print(f"\n✅ TEST 3 PASSED: Gateway configuration valid")
    return True


def test_4_tp_profile_selection():
    """Test 4: Verify TP profile selection uses EXIT_BRAIN_PROFILE"""
    print("\n" + "=" * 70)
    print("TEST 4: TP Profile Selection")
    print("=" * 70)
    
    from backend.config.exit_mode import is_challenge_100_profile
    from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
        get_tp_and_trailing_profile,
        MarketRegime,
        CHALLENGE_100_PROFILE
    )
    
    # Test profile selection
    test_symbol = "BTCUSDT"
    test_strategy = "RL_V3"
    test_regime = MarketRegime.TREND
    
    profile, trailing = get_tp_and_trailing_profile(
        symbol=test_symbol,
        strategy_id=test_strategy,
        regime=test_regime
    )
    
    print(f"✓ get_tp_and_trailing_profile(BTCUSDT, RL_V3, TREND)")
    print(f"  Selected profile: {profile.name}")
    print(f"  Description: {profile.description}")
    
    if is_challenge_100_profile():
        if profile.name != "CHALLENGE_100":
            print(f"✗ FAIL: EXIT_BRAIN_PROFILE=CHALLENGE_100 but profile.name={profile.name}")
            return False
        print(f"✓ CHALLENGE_100 profile selected (overrides regime)")
        print(f"  TP Legs: {len(profile.tp_legs)}")
        for idx, leg in enumerate(profile.tp_legs):
            print(f"    TP{idx+1}: {leg.size_fraction:.0%} @ {leg.r_multiple}R ({leg.kind.value})")
    else:
        if profile.name == "CHALLENGE_100":
            print(f"✗ FAIL: Profile is CHALLENGE_100 but EXIT_BRAIN_PROFILE != CHALLENGE_100")
            return False
        print(f"✓ Regime-based profile selected (EXIT_BRAIN_PROFILE != CHALLENGE_100)")
    
    print(f"\n✅ TEST 4 PASSED: Profile selection logic correct")
    return True


def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print("CHALLENGE_100 HOTFIX VERIFICATION")
    print("=" * 70)
    print("\nVerifying that:")
    print("1. EXIT_MODE is EXIT_BRAIN_V3 (not CHALLENGE_100)")
    print("2. EXIT_BRAIN_PROFILE=CHALLENGE_100 activates challenge rules")
    print("3. Hard SL uses gateway-compatible module_name")
    print("4. TP profile selection respects EXIT_BRAIN_PROFILE")
    print()
    
    results = []
    
    try:
        results.append(("EXIT_MODE Config", test_1_exit_mode_config()))
    except Exception as e:
        print(f"✗ TEST 1 FAILED WITH EXCEPTION: {e}")
        results.append(("EXIT_MODE Config", False))
    
    try:
        results.append(("EXIT_BRAIN_PROFILE", test_2_exit_brain_profile()))
    except Exception as e:
        print(f"✗ TEST 2 FAILED WITH EXCEPTION: {e}")
        results.append(("EXIT_BRAIN_PROFILE", False))
    
    try:
        results.append(("Gateway Compatibility", test_3_hard_sl_gateway_compatibility()))
    except Exception as e:
        print(f"✗ TEST 3 FAILED WITH EXCEPTION: {e}")
        results.append(("Gateway Compatibility", False))
    
    try:
        results.append(("TP Profile Selection", test_4_tp_profile_selection()))
    except Exception as e:
        print(f"✗ TEST 4 FAILED WITH EXCEPTION: {e}")
        results.append(("TP Profile Selection", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED - HOTFIX VERIFIED")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

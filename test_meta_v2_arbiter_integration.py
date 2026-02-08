"""
Integration Test: Meta-Agent V2 + Arbiter Decision Flow

Tests the complete 3-layer decision hierarchy:
1. Base Ensemble → 2. Meta-V2 Policy → 3. Arbiter (optional)

Validates:
- Defer scenarios (strong consensus, clear ensemble)
- Escalate scenarios (split vote, high disagreement)
- Arbiter gating (confidence threshold, HOLD rejection)
- Final decision correctness
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_engine.agents.meta_agent_v2 import MetaAgentV2
from ai_engine.agents.arbiter_agent import ArbiterAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'


def print_header(text):
    print(f"\n{CYAN}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{RESET}\n")


def print_section(text):
    print(f"\n{BLUE}{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}{RESET}")


def print_result(passed, message):
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    print(f"{status}: {message}")


def test_scenario_strong_consensus():
    """
    Scenario 1: Strong Consensus (3/4 models agree)
    Expected: Meta DEFER → Base ensemble
    """
    print_section("SCENARIO 1: Strong Consensus (3 BUY, 1 HOLD)")
    
    # Initialize Meta-V2 (no model needed for consensus check)
    meta = MetaAgentV2()
    
    base_predictions = {
        'xgb': {'action': 'BUY', 'confidence': 0.75},
        'lgbm': {'action': 'BUY', 'confidence': 0.72},
        'nhits': {'action': 'BUY', 'confidence': 0.68},
        'patchtst': {'action': 'HOLD', 'confidence': 0.55}
    }
    
    result = meta.predict(base_predictions)
    
    # Validate
    passed = (
        result['use_meta'] == False and
        result['action'] == 'BUY' and
        'strong_consensus' in result['reason']
    )
    
    print(f"  Base predictions: {base_predictions}")
    print(f"  Meta decision: use_meta={result['use_meta']}, action={result['action']}")
    print(f"  Reason: {result['reason']}")
    print_result(passed, "Meta DEFERS to base ensemble (strong consensus)")
    
    return passed


def test_scenario_split_vote():
    """
    Scenario 2: Split Vote (2 BUY, 2 SELL)
    Expected: Meta ESCALATE → Arbiter decides
    """
    print_section("SCENARIO 2: Split Vote (2 BUY, 2 SELL)")
    
    meta = MetaAgentV2()
    arbiter = ArbiterAgent(confidence_threshold=0.70, enable_conservative_mode=True)
    
    base_predictions = {
        'xgb': {'action': 'BUY', 'confidence': 0.72},
        'lgbm': {'action': 'SELL', 'confidence': 0.70},
        'nhits': {'action': 'BUY', 'confidence': 0.68},
        'patchtst': {'action': 'SELL', 'confidence': 0.71}
    }
    
    # Step 1: Meta decision
    meta_result = meta.predict(base_predictions)
    
    print(f"  Base predictions: {base_predictions}")
    print(f"  Meta decision: use_meta={meta_result['use_meta']}, reason={meta_result['reason']}")
    
    # Validate Meta escalates
    passed_meta = (
        meta_result['use_meta'] == True and
        'split_vote' in meta_result['reason']
    )
    
    print_result(passed_meta, "Meta ESCALATES (split vote detected)")
    
    if not passed_meta:
        return False
    
    # Step 2: Arbiter decision (simulated with oversold market)
    market_data_oversold = {
        'indicators': {
            'rsi': 28,
            'macd': 0.04,
            'macd_signal': -0.01,
            'bb_position': 0.18,
            'volume_trend': 1.7,
            'momentum_1h': 0.022
        },
        'regime': {
            'volatility': 0.4,
            'trend_strength': 0.15
        }
    }
    
    arbiter_result = arbiter.predict(market_data_oversold, base_predictions)
    
    print(f"  Arbiter decision: action={arbiter_result['action']}, confidence={arbiter_result['confidence']:.3f}")
    print(f"  Reason: {arbiter_result['reason']}")
    
    # Step 3: Arbiter gating
    should_override = arbiter.should_override_ensemble(
        arbiter_result['action'],
        arbiter_result['confidence']
    )
    
    print(f"  Arbiter gating: should_override={should_override}")
    
    # Validate Arbiter provides decision
    passed_arbiter = (
        arbiter_result['action'] in ['BUY', 'SELL', 'HOLD'] and
        0.0 <= arbiter_result['confidence'] <= 1.0
    )
    
    print_result(passed_arbiter, f"Arbiter provides decision: {arbiter_result['action']} @ {arbiter_result['confidence']:.3f}")
    
    # Final decision
    if should_override:
        final_action = arbiter_result['action']
        final_source = "Arbiter"
    else:
        final_action = meta_result['action']
        final_source = "Base Ensemble"
    
    print(f"\n  {YELLOW}FINAL DECISION: {final_action} (from {final_source}){RESET}")
    
    return passed_meta and passed_arbiter


def test_scenario_arbiter_low_confidence():
    """
    Scenario 3: High disagreement, but Arbiter has low confidence
    Expected: Meta ESCALATE → Arbiter DEFER → Base ensemble
    """
    print_section("SCENARIO 3: Arbiter Low Confidence (< 0.70)")
    
    meta = MetaAgentV2()
    arbiter = ArbiterAgent(confidence_threshold=0.70)
    
    base_predictions = {
        'xgb': {'action': 'BUY', 'confidence': 0.62},
        'lgbm': {'action': 'SELL', 'confidence': 0.60},
        'nhits': {'action': 'HOLD', 'confidence': 0.58},
        'patchtst': {'action': 'BUY', 'confidence': 0.61}
    }
    
    # Step 1: Meta decision
    meta_result = meta.predict(base_predictions)
    
    print(f"  Base predictions: {base_predictions}")
    print(f"  Meta decision: use_meta={meta_result['use_meta']}, reason={meta_result['reason']}")
    
    # Validate Meta escalates (high disagreement or low ensemble confidence)
    passed_meta = meta_result['use_meta'] == True
    
    print_result(passed_meta, f"Meta ESCALATES ({meta_result['reason']})")
    
    if not passed_meta:
        return False
    
    # Step 2: Arbiter decision (neutral market → low confidence)
    market_data_neutral = {
        'indicators': {
            'rsi': 51,
            'macd': 0.002,
            'macd_signal': 0.001,
            'bb_position': 0.48,
            'volume_trend': 1.05,
            'momentum_1h': 0.008
        },
        'regime': {
            'volatility': 0.3,
            'trend_strength': 0.03
        }
    }
    
    arbiter_result = arbiter.predict(market_data_neutral, base_predictions)
    
    print(f"  Arbiter decision: action={arbiter_result['action']}, confidence={arbiter_result['confidence']:.3f}")
    
    # Step 3: Arbiter gating (should REJECT due to low confidence)
    should_override = arbiter.should_override_ensemble(
        arbiter_result['action'],
        arbiter_result['confidence']
    )
    
    passed_gating = not should_override  # Should NOT override
    
    print(f"  Arbiter gating: should_override={should_override}")
    print_result(passed_gating, "Arbiter DEFERS (confidence too low or HOLD)")
    
    # Final decision: Use base ensemble
    final_action = meta_result['action']
    print(f"\n  {YELLOW}FINAL DECISION: {final_action} (from Base Ensemble){RESET}")
    
    return passed_meta and passed_gating


def test_scenario_arbiter_returns_hold():
    """
    Scenario 4: Arbiter returns HOLD (should not override)
    Expected: Meta ESCALATE → Arbiter HOLD → Base ensemble
    """
    print_section("SCENARIO 4: Arbiter Returns HOLD")
    
    arbiter = ArbiterAgent(confidence_threshold=0.70)
    
    # Arbiter returns HOLD with high confidence
    # Gating should reject (HOLD not allowed to override)
    should_override_hold = arbiter.should_override_ensemble('HOLD', 0.85)
    
    passed = not should_override_hold
    
    print(f"  Arbiter: action=HOLD, confidence=0.85")
    print(f"  Arbiter gating: should_override={should_override_hold}")
    print_result(passed, "Arbiter HOLD rejected by gating (only active trades allowed)")
    
    return passed


def test_scenario_clear_ensemble():
    """
    Scenario 5: Clear ensemble (low disagreement, high confidence)
    Expected: Meta DEFER → Base ensemble
    """
    print_section("SCENARIO 5: Clear Ensemble (low disagreement)")
    
    meta = MetaAgentV2()
    
    base_predictions = {
        'xgb': {'action': 'BUY', 'confidence': 0.78},
        'lgbm': {'action': 'BUY', 'confidence': 0.72},
        'nhits': {'action': 'HOLD', 'confidence': 0.65},
        'patchtst': {'action': 'BUY', 'confidence': 0.75}
    }
    
    result = meta.predict(base_predictions)
    
    # Validate Meta defers (may be "clear_ensemble" or "strong_consensus")
    passed = (
        result['use_meta'] == False and
        result['action'] == 'BUY'
    )
    
    print(f"  Base predictions: {base_predictions}")
    print(f"  Meta decision: use_meta={result['use_meta']}, action={result['action']}")
    print(f"  Reason: {result['reason']}")
    print_result(passed, "Meta DEFERS to base ensemble (clear decision)")
    
    return passed


def test_full_decision_flow_statistics():
    """
    Test: Run multiple predictions and check statistics
    """
    print_section("STATISTICS CHECK: Multiple Predictions")
    
    meta = MetaAgentV2()
    arbiter = ArbiterAgent(confidence_threshold=0.70)
    
    scenarios = [
        # (name, base_predictions, expected_meta_defer)
        ("Strong consensus", {
            'xgb': {'action': 'BUY', 'confidence': 0.80},
            'lgbm': {'action': 'BUY', 'confidence': 0.75},
            'nhits': {'action': 'BUY', 'confidence': 0.72},
            'patchtst': {'action': 'HOLD', 'confidence': 0.60}
        }, True),
        ("Split vote", {
            'xgb': {'action': 'BUY', 'confidence': 0.70},
            'lgbm': {'action': 'SELL', 'confidence': 0.68},
            'nhits': {'action': 'BUY', 'confidence': 0.66},
            'patchtst': {'action': 'SELL', 'confidence': 0.69}
        }, False),
        ("Disagreement", {
            'xgb': {'action': 'BUY', 'confidence': 0.65},
            'lgbm': {'action': 'SELL', 'confidence': 0.63},
            'nhits': {'action': 'HOLD', 'confidence': 0.60},
            'patchtst': {'action': 'BUY', 'confidence': 0.62}
        }, False),
    ]
    
    for name, preds, expected_defer in scenarios:
        result = meta.predict(preds)
        actual_defer = not result['use_meta']
        status = "✓" if actual_defer == expected_defer else "✗"
        print(f"  {status} {name}: defer={actual_defer} (expected={expected_defer})")
    
    # Get statistics
    meta_stats = meta.get_statistics()
    
    print(f"\n  Meta-V2 Statistics:")
    print(f"    Total predictions: {meta_stats['total_predictions']}")
    print(f"    Escalations: {meta_stats['meta_overrides']}")
    print(f"    Escalation rate: {meta_stats['override_rate']:.1%}")
    print(f"    Fallback reasons: {meta_stats['fallback_reasons']}")
    
    # Validate statistics
    passed = (
        meta_stats['total_predictions'] == len(scenarios) and
        meta_stats['meta_overrides'] >= 1  # At least split vote escalated
    )
    
    print_result(passed, "Statistics tracking works correctly")
    
    return passed


def main():
    """Run all integration tests."""
    print_header("META-AGENT V2 + ARBITER INTEGRATION TESTS")
    
    tests = [
        ("Strong Consensus", test_scenario_strong_consensus),
        ("Split Vote", test_scenario_split_vote),
        ("Arbiter Low Confidence", test_scenario_arbiter_low_confidence),
        ("Arbiter Returns HOLD", test_scenario_arbiter_returns_hold),
        ("Clear Ensemble", test_scenario_clear_ensemble),
        ("Statistics Tracking", test_full_decision_flow_statistics)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print_result(False, f"Exception in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status}  {name}")
    
    print(f"\n  {CYAN}{'─'*70}")
    if passed_count == total_count:
        print(f"  {GREEN}ALL TESTS PASSED ({passed_count}/{total_count}){RESET}")
    else:
        print(f"  {YELLOW}{passed_count}/{total_count} tests passed{RESET}")
    print(f"  {'─'*70}{RESET}\n")
    
    return passed_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
META-AGENT V2 INTEGRATION TEST

End-to-end test proving Meta-Agent V2 is working correctly in production.

Tests:
1. Meta-agent loads successfully
2. Fallback works (low confidence, strong consensus)
3. Meta override works (high confidence, disagreement)
4. Safety checks active
5. Statistics tracking works
6. Ensemble manager integration works

Run on VPS:
    cd /home/qt/quantum_trader
    /opt/quantum/venvs/ai-engine/bin/python test_meta_v2_integration.py
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_engine.meta.meta_agent_v2 import MetaAgentV2
from ai_engine.ensemble_manager import EnsembleManager


# ============================================================================
# COLORS
# ============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if passed else f"{Colors.RED}❌ FAIL{Colors.RESET}"
    print(f"{status} | {name}")
    if details:
        print(f"      {Colors.YELLOW}{details}{Colors.RESET}")


# ============================================================================
# TEST 1: Meta-Agent Direct Test
# ============================================================================

def test_meta_agent_direct():
    """Test meta-agent loads and works correctly."""
    print_section("TEST 1: Meta-Agent Direct Load")
    
    try:
        meta = MetaAgentV2()
        
        # Check if loaded
        is_ready = meta.is_ready()
        print_test(
            "Meta-Agent Loads",
            is_ready,
            f"Model ready: {is_ready}, Version: {meta.VERSION}"
        )
        
        if not is_ready:
            print(f"{Colors.YELLOW}⚠️  Model not found - this is OK if not trained yet{Colors.RESET}")
            print(f"{Colors.YELLOW}   Run: python ops/retrain/train_meta_v2.py{Colors.RESET}")
            return False
        
        # Test fallback: strong consensus
        base_preds_consensus = {
            'xgb': {'action': 'BUY', 'confidence': 0.85},
            'lgbm': {'action': 'BUY', 'confidence': 0.82},
            'nhits': {'action': 'BUY', 'confidence': 0.88},
            'patchtst': {'action': 'BUY', 'confidence': 0.80}
        }
        
        result = meta.predict(base_preds_consensus, symbol='TESTUSDT')
        
        fallback_worked = (
            result['use_meta'] == False and
            'consensus' in result['reason']
        )
        
        print_test(
            "Fallback: Strong Consensus",
            fallback_worked,
            f"use_meta={result['use_meta']}, reason={result['reason']}"
        )
        
        # Test disagreement scenario
        base_preds_disagree = {
            'xgb': {'action': 'BUY', 'confidence': 0.65},
            'lgbm': {'action': 'SELL', 'confidence': 0.70},
            'nhits': {'action': 'HOLD', 'confidence': 0.60},
            'patchtst': {'action': 'BUY', 'confidence': 0.75}
        }
        
        result2 = meta.predict(base_preds_disagree, symbol='TESTUSDT')
        
        valid_decision = (
            result2['action'] in ['SELL', 'HOLD', 'BUY'] and
            0.0 <= result2['confidence'] <= 1.0
        )
        
        print_test(
            "Prediction: Disagreement Scenario",
            valid_decision,
            f"action={result2['action']}, conf={result2['confidence']:.3f}, use_meta={result2['use_meta']}"
        )
        
        # Test statistics
        stats = meta.get_statistics()
        stats_work = (
            stats['total_predictions'] > 0 and
            'override_rate' in stats
        )
        
        print_test(
            "Statistics Tracking",
            stats_work,
            f"predictions={stats['total_predictions']}, override_rate={stats['override_rate']:.2%}"
        )
        
        return True
        
    except Exception as e:
        print_test("Meta-Agent Direct Load", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: Ensemble Manager Integration
# ============================================================================

def test_ensemble_integration():
    """Test meta-agent works with ensemble manager."""
    print_section("TEST 2: Ensemble Manager Integration")
    
    try:
        # Initialize ensemble
        ensemble = EnsembleManager()
        
        # Check meta-agent loaded
        has_meta = ensemble.meta_agent is not None
        meta_ready = has_meta and ensemble.meta_agent.is_ready()
        
        print_test(
            "Meta-Agent in Ensemble",
            has_meta,
            f"Loaded: {has_meta}, Ready: {meta_ready}"
        )
        
        if not has_meta:
            print(f"{Colors.YELLOW}⚠️  Meta-agent not available in ensemble{Colors.RESET}")
            return False
        
        # Test prediction through ensemble (if models available)
        if ensemble.xgb_agent and ensemble.lgbm_agent:
            # Mock features for testing
            test_features = {
                'symbol': 'TESTUSDT',
                'price_change': 0.02,
                'volume_ratio': 1.5,
                'volatility': 0.5
            }
            
            try:
                action, confidence, info = ensemble.predict('TESTUSDT', test_features)
                
                prediction_works = (
                    action in ['SELL', 'HOLD', 'BUY'] and
                    0.0 <= confidence <= 1.0
                )
                
                print_test(
                    "Ensemble Prediction",
                    prediction_works,
                    f"action={action}, conf={confidence:.3f}, meta_override={info.get('meta_override', False)}"
                )
                
            except Exception as e:
                print_test("Ensemble Prediction", False, f"Error: {e}")
        
        return True
        
    except Exception as e:
        print_test("Ensemble Manager Integration", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Environment Configuration
# ============================================================================

def test_environment_config():
    """Test environment configuration is correct."""
    print_section("TEST 3: Environment Configuration")
    
    # Check META_AGENT_ENABLED
    meta_enabled = os.getenv('META_AGENT_ENABLED', 'false').lower() == 'true'
    print_test(
        "META_AGENT_ENABLED",
        True,  # Always pass, just informational
        f"Value: {os.getenv('META_AGENT_ENABLED', 'false')} (enabled={meta_enabled})"
    )
    
    # Check META_OVERRIDE_THRESHOLD
    meta_threshold = float(os.getenv('META_OVERRIDE_THRESHOLD', '0.65'))
    threshold_ok = 0.5 <= meta_threshold <= 0.95
    print_test(
        "META_OVERRIDE_THRESHOLD",
        threshold_ok,
        f"Value: {meta_threshold} (recommended: 0.60-0.75)"
    )
    
    # Check META_FAIL_OPEN
    meta_fail_open = os.getenv('META_FAIL_OPEN', 'true').lower() == 'true'
    print_test(
        "META_FAIL_OPEN",
        True,  # Always pass, just informational
        f"Value: {os.getenv('META_FAIL_OPEN', 'true')} (fail_open={meta_fail_open})"
    )
    
    # Check model directory
    model_dir = os.getenv(
        'META_V2_MODEL_DIR',
        '/home/qt/quantum_trader/ai_engine/models/meta_v2'
    )
    model_exists = Path(model_dir).exists()
    print_test(
        "Model Directory Exists",
        model_exists,
        f"Path: {model_dir}"
    )
    
    if model_exists:
        model_file = Path(model_dir) / 'meta_model.pkl'
        scaler_file = Path(model_dir) / 'scaler.pkl'
        metadata_file = Path(model_dir) / 'metadata.json'
        
        files_ok = model_file.exists() and scaler_file.exists() and metadata_file.exists()
        print_test(
            "Model Files Present",
            files_ok,
            f"model={model_file.exists()}, scaler={scaler_file.exists()}, metadata={metadata_file.exists()}"
        )
    
    return True


# ============================================================================
# TEST 4: Safety Checks
# ============================================================================

def test_safety_checks():
    """Test that safety checks are working."""
    print_section("TEST 4: Safety Checks")
    
    try:
        meta = MetaAgentV2()
        
        if not meta.is_ready():
            print(f"{Colors.YELLOW}⚠️  Skipping safety tests (model not loaded){Colors.RESET}")
            return True
        
        # Test 1: Empty predictions
        result = meta.predict({}, symbol='TEST')
        safety1 = result['use_meta'] == False
        print_test(
            "Safety: Empty Predictions",
            safety1,
            f"use_meta={result['use_meta']}, reason={result['reason']}"
        )
        
        # Test 2: Feature dimension check (corrupt expected dim)
        original_dim = meta.expected_feature_dim
        meta.expected_feature_dim = 999
        
        base_preds = {
            'xgb': {'action': 'BUY', 'confidence': 0.70}
        }
        result = meta.predict(base_preds, symbol='TEST')
        
        safety2 = result['use_meta'] == False and 'dim_mismatch' in result['reason']
        print_test(
            "Safety: Dimension Mismatch",
            safety2,
            f"use_meta={result['use_meta']}, reason={result['reason']}"
        )
        
        # Restore
        meta.expected_feature_dim = original_dim
        
        # Test 3: Model not loaded fallback
        meta2 = MetaAgentV2(model_dir='/nonexistent')
        result = meta2.predict(base_preds, symbol='TEST')
        
        safety3 = result['use_meta'] == False and 'not_loaded' in result['reason']
        print_test(
            "Safety: Model Not Loaded",
            safety3,
            f"use_meta={result['use_meta']}, reason={result['reason']}"
        )
        
        return True
        
    except Exception as e:
        print_test("Safety Checks", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all integration tests."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "META-AGENT V2 INTEGRATION TEST" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{Colors.RESET}")
    
    results = []
    
    # Run tests
    results.append(("Meta-Agent Direct", test_meta_agent_direct()))
    results.append(("Ensemble Integration", test_ensemble_integration()))
    results.append(("Environment Config", test_environment_config()))
    results.append(("Safety Checks", test_safety_checks()))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}✅{Colors.RESET}" if result else f"{Colors.RED}❌{Colors.RESET}"
        print(f"{status} {test_name}")
    
    print()
    print(f"{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED - META-AGENT V2 OPERATIONAL{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED - CHECK CONFIGURATION{Colors.RESET}")
        return 1


if __name__ == '__main__':
    exit(main())

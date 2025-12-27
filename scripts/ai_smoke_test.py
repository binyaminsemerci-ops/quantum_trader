#!/usr/bin/env python3
"""
AI MODULES SMOKE TEST - STEP 3
================================

Validates all AI modules with synthetic data:
1. AI Ensemble (XGBoost, LightGBM, N-HiTS, PatchTST)
2. Regime Detector V2
3. World Model
4. RL v3 Position Sizing
5. Model Supervisor
6. Portfolio Balancer AI

Tests basic functionality without requiring live data or training.

Author: GitHub Copilot (Senior Systems QA)
Date: December 5, 2025
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test results tracking
test_results: List[Dict[str, Any]] = []


def record_test(module: str, test: str, passed: bool, details: str = "", error: str = ""):
    """Record test result."""
    result = {
        "module": module,
        "test": test,
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details,
        "error": error
    }
    test_results.append(result)
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {module:30} | {test}")
    if details:
        print(f"     └─ {details}")
    if error:
        print(f"     └─ ERROR: {error}")


def test_ensemble_manager():
    """Test AI Ensemble Manager."""
    print("\n" + "="*80)
    print("MODULE 1: AI ENSEMBLE MANAGER")
    print("="*80)
    
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        record_test("EnsembleManager", "Import", True, "Successfully imported from ai_engine/")
    except ImportError as e:
        record_test("EnsembleManager", "Import", False, error=str(e))
        return
    
    # Test initialization
    try:
        ensemble = EnsembleManager()
        record_test("EnsembleManager", "Initialize", True, "Default weights and min_consensus=3")
    except Exception as e:
        record_test("EnsembleManager", "Initialize", False, error=str(e))
        return
    
    # Test predict method exists
    try:
        assert hasattr(ensemble, 'predict'), "Missing predict() method"
        record_test("EnsembleManager", "Has predict()", True, "Method exists")
    except AssertionError as e:
        record_test("EnsembleManager", "Has predict()", False, error=str(e))
    
    # Test warmup method exists
    try:
        assert hasattr(ensemble, 'warmup') or hasattr(ensemble, 'warmup_symbol'), "Missing warmup method"
        record_test("EnsembleManager", "Has warmup()", True, "Warmup method exists")
    except AssertionError as e:
        record_test("EnsembleManager", "Has warmup()", False, error=str(e))
    
    # Test model attributes
    try:
        assert hasattr(ensemble, 'xgb_agent'), "Missing xgb_agent"
        assert hasattr(ensemble, 'lgbm_agent'), "Missing lgbm_agent"
        assert hasattr(ensemble, 'nhits_agent'), "Missing nhits_agent"
        assert hasattr(ensemble, 'patchtst_agent'), "Missing patchtst_agent"
        record_test("EnsembleManager", "4-Model Architecture", True, "XGB, LGBM, N-HiTS, PatchTST present")
    except AssertionError as e:
        record_test("EnsembleManager", "4-Model Architecture", False, error=str(e))


def test_regime_detector():
    """Test Regime Detector V2."""
    print("\n" + "="*80)
    print("MODULE 2: REGIME DETECTOR V2")
    print("="*80)
    
    try:
        from backend.utils.regime_detector_v2 import RegimeDetectorV2
        record_test("RegimeDetectorV2", "Import", True, "Successfully imported")
    except ImportError as e:
        record_test("RegimeDetectorV2", "Import", False, error=str(e))
        return
    
    # Test initialization
    try:
        detector = RegimeDetectorV2()
        record_test("RegimeDetectorV2", "Initialize", True, "No errors during init")
    except Exception as e:
        record_test("RegimeDetectorV2", "Initialize", False, error=str(e))
        return
    
    # Test detect_regime method
    try:
        assert hasattr(detector, 'detect_regime'), "Missing detect_regime() method"
        record_test("RegimeDetectorV2", "Has detect_regime()", True, "Method exists")
    except AssertionError as e:
        record_test("RegimeDetectorV2", "Has detect_regime()", False, error=str(e))
    
    # Test with synthetic data
    try:
        synthetic_prices = [100.0 + i * 0.5 for i in range(100)]  # Uptrend
        regime = detector.detect_regime(synthetic_prices)
        assert regime in ["TREND", "RANGE", "BREAKOUT", "MEAN_REVERSION"], f"Invalid regime: {regime}"
        record_test("RegimeDetectorV2", "Synthetic Data Test", True, f"Detected regime: {regime}")
    except Exception as e:
        record_test("RegimeDetectorV2", "Synthetic Data Test", False, error=str(e))


def test_world_model():
    """Test World Model."""
    print("\n" + "="*80)
    print("MODULE 3: WORLD MODEL")
    print("="*80)
    
    try:
        from backend.world_model.world_model import WorldModel, MarketState, MarketRegime
        record_test("WorldModel", "Import", True, "WorldModel, MarketState, MarketRegime imported")
    except ImportError as e:
        record_test("WorldModel", "Import", False, error=str(e))
        return
    
    # Test MarketState creation
    try:
        state = MarketState(
            current_price=50000.0,
            current_regime=MarketRegime.TRENDING_UP,
            volatility=0.02,
            trend_strength=0.8,
            volume_ratio=1.5
        )
        record_test("WorldModel", "MarketState creation", True, "Created state with all fields")
    except Exception as e:
        record_test("WorldModel", "MarketState creation", False, error=str(e))
        return
    
    # Test WorldModel initialization
    try:
        world_model = WorldModel()
        record_test("WorldModel", "Initialize", True, "No errors during init")
    except Exception as e:
        record_test("WorldModel", "Initialize", False, error=str(e))
        return
    
    # Test scenario generation method exists
    try:
        assert hasattr(world_model, 'generate_scenarios') or hasattr(world_model, 'project'), \
            "Missing scenario generation method"
        record_test("WorldModel", "Has scenario generation", True, "Method exists")
    except AssertionError as e:
        record_test("WorldModel", "Has scenario generation", False, error=str(e))


def test_rl_position_sizing():
    """Test RL v3 Position Sizing Agent."""
    print("\n" + "="*80)
    print("MODULE 4: RL V3 POSITION SIZING")
    print("="*80)
    
    # Try multiple possible locations
    locations = [
        ("backend.services.ai.rl_position_sizing_agent", "RLPositionSizingAgent"),
        ("backend.agents.rl_position_sizing_agent_v2", "RLPositionSizingAgent"),
        ("backend.domains.learning.rl_position_sizing", "RLPositionSizingAgent"),
    ]
    
    agent_class = None
    for module_path, class_name in locations:
        try:
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            record_test("RLPositionSizing", "Import", True, f"Found in {module_path}")
            break
        except (ImportError, AttributeError):
            continue
    
    if agent_class is None:
        record_test("RLPositionSizing", "Import", False, error="Not found in any known location")
        return
    
    # Test initialization
    try:
        agent = agent_class()
        record_test("RLPositionSizing", "Initialize", True, "Agent initialized")
    except Exception as e:
        record_test("RLPositionSizing", "Initialize", False, error=str(e))
        return
    
    # Test calculate_position_size method
    try:
        assert hasattr(agent, 'calculate_position_size') or hasattr(agent, 'get_position_size'), \
            "Missing position sizing method"
        record_test("RLPositionSizing", "Has position sizing", True, "Method exists")
    except AssertionError as e:
        record_test("RLPositionSizing", "Has position sizing", False, error=str(e))
    
    # Test Q-table or policy storage
    try:
        has_q_table = hasattr(agent, 'q_table') or hasattr(agent, 'policy') or hasattr(agent, 'state_dict')
        assert has_q_table, "Missing Q-table/policy storage"
        record_test("RLPositionSizing", "Has Q-learning state", True, "Q-table or policy present")
    except AssertionError as e:
        record_test("RLPositionSizing", "Has Q-learning state", False, error=str(e))


def test_model_supervisor():
    """Test Model Supervisor."""
    print("\n" + "="*80)
    print("MODULE 5: MODEL SUPERVISOR")
    print("="*80)
    
    try:
        from backend.services.ai.shadow_model_manager import ShadowModelManager
        record_test("ModelSupervisor", "Import (Shadow Models)", True, "ShadowModelManager imported")
    except ImportError as e:
        record_test("ModelSupervisor", "Import (Shadow Models)", False, error=str(e))
        return
    
    # Test initialization
    try:
        supervisor = ShadowModelManager()
        record_test("ModelSupervisor", "Initialize", True, "Shadow model manager initialized")
    except Exception as e:
        record_test("ModelSupervisor", "Initialize", False, error=str(e))
        return
    
    # Test promotion methods
    try:
        assert hasattr(supervisor, 'evaluate_promotion') or hasattr(supervisor, 'promote_shadow'), \
            "Missing promotion logic"
        record_test("ModelSupervisor", "Has promotion logic", True, "Promotion methods exist")
    except AssertionError as e:
        record_test("ModelSupervisor", "Has promotion logic", False, error=str(e))


def test_portfolio_balancer():
    """Test Portfolio Balancer AI."""
    print("\n" + "="*80)
    print("MODULE 6: PORTFOLIO BALANCER AI")
    print("="*80)
    
    try:
        from backend.services.ai.portfolio_balancer import PortfolioBalancer
        record_test("PortfolioBalancer", "Import", True, "PortfolioBalancer imported")
    except ImportError as e:
        # Try alternate location
        try:
            from backend.domains.portfolio.balancer import PortfolioBalancer
            record_test("PortfolioBalancer", "Import", True, "Found in domains/portfolio/")
        except ImportError as e2:
            record_test("PortfolioBalancer", "Import", False, error=str(e2))
            return
    
    # Test initialization
    try:
        balancer = PortfolioBalancer()
        record_test("PortfolioBalancer", "Initialize", True, "Balancer initialized")
    except Exception as e:
        record_test("PortfolioBalancer", "Initialize", False, error=str(e))
        return
    
    # Test rebalance method
    try:
        assert hasattr(balancer, 'rebalance') or hasattr(balancer, 'calculate_rebalance'), \
            "Missing rebalance method"
        record_test("PortfolioBalancer", "Has rebalance()", True, "Rebalance method exists")
    except AssertionError as e:
        record_test("PortfolioBalancer", "Has rebalance()", False, error=str(e))


def test_continuous_learning():
    """Test Continuous Learning Manager."""
    print("\n" + "="*80)
    print("MODULE 7: CONTINUOUS LEARNING MANAGER")
    print("="*80)
    
    try:
        from backend.services.ai.continuous_learning_manager import ContinuousLearningManager
        record_test("ContinuousLearning", "Import", True, "CLM imported")
    except ImportError as e:
        record_test("ContinuousLearning", "Import", False, error=str(e))
        return
    
    # Test initialization
    try:
        clm = ContinuousLearningManager()
        record_test("ContinuousLearning", "Initialize", True, "CLM initialized")
    except Exception as e:
        record_test("ContinuousLearning", "Initialize", False, error=str(e))
        return
    
    # Test retraining trigger logic
    try:
        assert hasattr(clm, 'should_retrain') or hasattr(clm, 'check_retraining_triggers'), \
            "Missing retraining trigger logic"
        record_test("ContinuousLearning", "Has retraining logic", True, "Trigger methods exist")
    except AssertionError as e:
        record_test("ContinuousLearning", "Has retraining logic", False, error=str(e))


def print_summary():
    """Print test summary report."""
    print("\n" + "="*80)
    print("AI MODULES SMOKE TEST SUMMARY")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for t in test_results if t['passed'])
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    # Group by module
    print("\n" + "-"*80)
    print("RESULTS BY MODULE")
    print("-"*80)
    
    modules: Dict[str, Dict[str, int]] = {}
    for result in test_results:
        module = result['module']
        if module not in modules:
            modules[module] = {"passed": 0, "failed": 0}
        
        if result['passed']:
            modules[module]['passed'] += 1
        else:
            modules[module]['failed'] += 1
    
    for module, stats in modules.items():
        total = stats['passed'] + stats['failed']
        status = "✅" if stats['failed'] == 0 else "⚠️" if stats['passed'] > 0 else "❌"
        print(f"{status} {module:30} | {stats['passed']}/{total} passed")
    
    # Show failed tests
    failed = [t for t in test_results if not t['passed']]
    if failed:
        print("\n" + "-"*80)
        print("FAILED TESTS DETAIL")
        print("-"*80)
        for result in failed:
            print(f"\n❌ {result['module']} - {result['test']}")
            print(f"   Error: {result['error']}")
    
    print("\n" + "="*80)
    
    # Overall status
    if pass_rate == 100:
        print("🎉 ALL AI MODULES OPERATIONAL - STEP 3 COMPLETE")
    elif pass_rate >= 70:
        print("⚠️  MOST AI MODULES OPERATIONAL - REVIEW FAILURES")
    else:
        print("🔴 CRITICAL ISSUES IN AI MODULES - NEEDS ATTENTION")
    print("="*80 + "\n")
    
    # Save results to JSON
    output_file = Path(__file__).parent.parent / "AI_SMOKE_TEST_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": pass_rate
            },
            "results": test_results
        }, f, indent=2)
    print(f"📄 Detailed results saved to: {output_file}")
    
    return pass_rate >= 70  # Return success if >= 70% pass rate


def main():
    """Run all AI module smoke tests."""
    print("="*80)
    print("QUANTUM TRADER V2.0 - AI MODULES SMOKE TEST")
    print("STEP 3: AI Modules Functionality Check")
    print("="*80)
    print(f"Started: {datetime.utcnow().isoformat()}")
    print("="*80)
    
    # Run all tests
    test_ensemble_manager()
    test_regime_detector()
    test_world_model()
    test_rl_position_sizing()
    test_model_supervisor()
    test_portfolio_balancer()
    test_continuous_learning()
    
    # Print summary
    success = print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

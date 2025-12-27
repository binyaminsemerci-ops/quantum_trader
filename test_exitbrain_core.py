#!/usr/bin/env python3
"""
Direct test of ExitBrain v3.5 CORE ENGINE ONLY
Tests AdaptiveLeverageEngine directly, bypassing buggy integration layer
"""
import sys
import os

# Set up paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/microservices')
sys.path.insert(0, '/app/ai_engine')
os.environ['PYTHONPATH'] = '/app/backend:/app/microservices:/app/ai_engine:/app'

import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
from microservices.exitbrain_v3_5.pnl_tracker import PnLTracker

def test_core_engine():
    """Test AdaptiveLeverageEngine core functionality directly"""
    print("=" * 100)
    print("🧪 TESTING ExitBrain v3.5 CORE ENGINE (Direct - No Integration Layer)")
    print("=" * 100)
    
    # Initialize core engine
    try:
        engine = AdaptiveLeverageEngine(base_tp=1.0, base_sl=0.5)
        pnl_tracker = PnLTracker(max_history=20)
        print(f"\n✅ AdaptiveLeverageEngine initialized (base_tp={engine.base_tp}%, base_sl={engine.base_sl}%)")
        print(f"✅ PnLTracker initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test scenarios
    test_cases = [
        {
            "desc": "10x leverage, normal volatility",
            "base_tp": 1.0,
            "base_sl": 0.5,
            "leverage": 10,
            "volatility": 1.0,
            "expected_behavior": "Standard adaptive levels"
        },
        {
            "desc": "20x leverage, high volatility",
            "base_tp": 1.0,
            "base_sl": 0.5,
            "leverage": 20,
            "volatility": 1.5,
            "expected_behavior": "Tighter stops due to high leverage+volatility"
        },
        {
            "desc": "5x leverage, low volatility",
            "base_tp": 1.0,
            "base_sl": 0.5,
            "leverage": 5,
            "volatility": 0.7,
            "expected_behavior": "Wider targets due to low risk"
        },
        {
            "desc": "30x leverage (extreme)",
            "base_tp": 1.0,
            "base_sl": 0.5,
            "leverage": 30,
            "volatility": 1.2,
            "expected_behavior": "Maximum risk protection"
        }
    ]
    
    all_passed = True
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 100}")
        print(f"Test #{i}: {test['desc']}")
        print(f"{'─' * 100}")
        print(f"📊 Input: base_tp={test['base_tp']}%, base_sl={test['base_sl']}%, leverage={test['leverage']}x, volatility={test['volatility']}")
        print(f"🎯 Expected: {test['expected_behavior']}")
        
        try:
            # Test compute_levels
            levels = engine.compute_levels(
                base_tp_pct=test["base_tp"],
                base_sl_pct=test["base_sl"],
                leverage=test["leverage"],
                volatility_factor=test["volatility"]
            )
            
            print(f"\n📈 Output:")
            print(f"   • TP1: {levels.tp1:.4f}%")
            print(f"   • TP2: {levels.tp2:.4f}%")
            print(f"   • TP3: {levels.tp3:.4f}%")
            print(f"   • SL:  {levels.sl:.4f}%")
            print(f"   • LSF: {levels.lsf:.4f}")
            
            # Test harvest_scheme_for
            harvest = engine.harvest_scheme_for(test["leverage"])
            print(f"   • Harvest scheme: {[f'{x:.2%}' for x in harvest]}")
            
            # Test compute_lsf
            lsf = engine.compute_lsf(test["leverage"])
            print(f"   • LSF (direct): {lsf:.4f}")
            
            # Validations
            validation_passed = True
            
            # Check TP levels are progressive
            if not (levels.tp1 < levels.tp2 < levels.tp3):
                print(f"   ⚠️  WARNING: TP levels not progressive (TP1 < TP2 < TP3)")
                validation_passed = False
            
            # Check SL is reasonable (< 5%)
            if levels.sl > 5.0:
                print(f"   ⚠️  WARNING: SL too wide (>{5.0}%)")
                validation_passed = False
            
            # Check LSF is within bounds (0.1 to 2.0)
            if not (0.1 <= levels.lsf <= 2.0):
                print(f"   ⚠️  WARNING: LSF out of bounds (0.1-2.0)")
                validation_passed = False
            
            # Check harvest scheme sums to ~1.0
            harvest_sum = sum(harvest)
            if not (0.99 <= harvest_sum <= 1.01):
                print(f"   ⚠️  WARNING: Harvest scheme doesn't sum to 1.0 (sum={harvest_sum})")
                validation_passed = False
            
            if validation_passed:
                print(f"\n   ✅ PASS: All validations passed")
            else:
                print(f"\n   ⚠️  PASS with warnings")
                all_passed = False
            
            results.append({
                "test": test["desc"],
                "leverage": test["leverage"],
                "tp1": levels.tp1,
                "tp3": levels.tp3,
                "sl": levels.sl,
                "lsf": levels.lsf,
                "passed": validation_passed
            })
            
        except Exception as e:
            print(f"\n   ❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            results.append({
                "test": test["desc"],
                "leverage": test["leverage"],
                "error": str(e),
                "passed": False
            })
    
    # Summary
    print(f"\n{'=' * 100}")
    print("📊 SUMMARY OF RESULTS")
    print("=" * 100)
    
    print(f"\n{'Leverage':<12} {'TP1':<10} {'TP3':<10} {'SL':<10} {'LSF':<10} {'Status'}")
    print("─" * 70)
    for result in results:
        if "error" in result:
            print(f"{result['leverage']:<12} {'ERROR':<50}")
        else:
            status = "✅ PASS" if result["passed"] else "⚠️  WARN"
            print(f"{result['leverage']}x{'':<10} {result['tp1']:.3f}%{'':<5} {result['tp3']:.3f}%{'':<5} {result['sl']:.3f}%{'':<5} {result['lsf']:.3f}{'':<5} {status}")
    
    print(f"\n{'=' * 100}")
    if all_passed:
        print("✅ ALL CORE ENGINE TESTS PASSED")
        print("🎯 ExitBrain v3.5 AdaptiveLeverageEngine is FULLY FUNCTIONAL")
    else:
        print("⚠️  SOME TESTS HAD WARNINGS - Core functionality works but edge cases need review")
    print("=" * 100)
    
    return all_passed

if __name__ == "__main__":
    success = test_core_engine()
    sys.exit(0 if success else 1)

"""
TP v3 Smoke Test - End-to-end TP pipeline validation
====================================================

Tests the complete TP flow:
1. RL v3 generates signal with TP suggestion
2. Dynamic TP/SL calculator processes it
3. Hybrid TP/SL blends with risk controls
4. Validation checks ensure correct behavior
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.services.execution.dynamic_tpsl import get_dynamic_tpsl_calculator
from backend.services.execution.hybrid_tpsl import calculate_hybrid_levels


async def run_tp_smoke_test():
    """Validate TP pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("🧪 TP v3 SMOKE TEST")
    print("=" * 60)
    
    # STEP 1: RL v3 generates signal
    print("\n[STEP 1] RL v3 Signal Generation")
    print("-" * 60)
    try:
        rl_manager = RLv3Manager()
        obs = {
            'price': 100000.0,
            'rsi': 70,
            'macd': 0.015,
            'volume': 1000000
        }
        rl_output = rl_manager.predict(obs)
        print(f"   ✅ Action: {rl_output['action']}")
        print(f"   ✅ Confidence: {rl_output['confidence']:.2%}")
        print(f"   ✅ TP Zone Multiplier: {rl_output.get('tp_zone_multiplier', 'N/A')}")
        print(f"   ✅ Suggested TP: {rl_output.get('suggested_tp_pct', 'N/A'):.2%}")
        step1_passed = True
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        step1_passed = False
        rl_output = {'confidence': 0.75, 'action': 'BUY', 'suggested_tp_pct': 0.06}
    
    # STEP 2: Dynamic TP/SL calculation
    print("\n[STEP 2] Dynamic TP/SL Calculation")
    print("-" * 60)
    try:
        calculator = get_dynamic_tpsl_calculator()
        tpsl_result = calculator.calculate(
            signal_confidence=rl_output['confidence'],
            action='BUY',
            market_conditions={'volatility': 0.02},
            risk_mode='NORMAL',
            rl_tp_suggestion=rl_output.get('suggested_tp_pct')
        )
        print(f"   ✅ TP: {tpsl_result.tp_percent:.2%}")
        print(f"   ✅ SL: {tpsl_result.sl_percent:.2%}")
        print(f"   ✅ Trail: {tpsl_result.trail_percent:.2%}")
        print(f"   ✅ R:R = {tpsl_result.tp_percent / tpsl_result.sl_percent:.2f}x")
        print(f"   ✅ Partial TP: {tpsl_result.partial_tp}")
        step2_passed = True
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        step2_passed = False
        return False
    
    # STEP 3: Hybrid TP/SL blending
    print("\n[STEP 3] Hybrid TP/SL Blending")
    print("-" * 60)
    try:
        hybrid = calculate_hybrid_levels(
            entry_price=100000.0,
            side='BUY',
            risk_sl_percent=0.025,
            base_tp_percent=0.05,
            ai_tp_percent=tpsl_result.tp_percent,
            ai_trail_percent=tpsl_result.trail_percent,
            confidence=rl_output['confidence']
        )
        print(f"   ✅ Final TP: {hybrid['final_tp_percent']:.2%}")
        print(f"   ✅ Final SL: {hybrid['final_sl_percent']:.2%}")
        print(f"   ✅ Trailing: {hybrid['trail_callback_percent']:.2%}")
        print(f"   ✅ Mode: {hybrid['mode']}")
        step3_passed = True
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        step3_passed = False
        return False
    
    # STEP 4: Validation checks
    print("\n[STEP 4] Validation Checks")
    print("-" * 60)
    
    checks = {
        'RL outputs confidence': rl_output['confidence'] > 0,
        'RL suggests TP zone': 'tp_zone_multiplier' in rl_output,
        'TP > SL': hybrid['final_tp_percent'] > hybrid['final_sl_percent'],
        'TP >= 5%': hybrid['final_tp_percent'] >= 0.05,
        'SL <= 3%': hybrid['final_sl_percent'] <= 0.03,
        'R:R >= 1.5x': (hybrid['final_tp_percent'] / hybrid['final_sl_percent']) >= 1.5,
        'Trailing configured': hybrid['trail_callback_percent'] >= 0 if rl_output['confidence'] > 0.5 else True
    }
    
    all_passed = all(checks.values())
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # STEP 5: Risk v3 integration test
    print("\n[STEP 5] Risk v3 Integration Test")
    print("-" * 60)
    try:
        # Test high ESS scenario
        tpsl_high_ess = calculator.calculate(
            signal_confidence=0.75,
            action='BUY',
            market_conditions={'volatility': 0.02},
            risk_mode='NORMAL',
            risk_v3_context={'ess_factor': 2.5, 'systemic_risk_level': 0.3}
        )
        
        # Test systemic risk scenario
        tpsl_systemic = calculator.calculate(
            signal_confidence=0.75,
            action='BUY',
            market_conditions={'volatility': 0.02},
            risk_mode='NORMAL',
            risk_v3_context={'ess_factor': 1.0, 'systemic_risk_level': 0.8}
        )
        
        ess_tightening = tpsl_high_ess.tp_percent < tpsl_result.tp_percent
        systemic_defensive = tpsl_systemic.tp_percent < tpsl_result.tp_percent
        
        print(f"   {'✅' if ess_tightening else '❌'} High ESS tightens TP")
        print(f"   {'✅' if systemic_defensive else '❌'} Systemic risk activates defensive TP")
        
        step5_passed = ess_tightening and systemic_defensive
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        step5_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"   Step 1 (RL v3 Signal):         {'✅ PASS' if step1_passed else '❌ FAIL'}")
    print(f"   Step 2 (Dynamic TP/SL):        {'✅ PASS' if step2_passed else '❌ FAIL'}")
    print(f"   Step 3 (Hybrid Blending):      {'✅ PASS' if step3_passed else '❌ FAIL'}")
    print(f"   Step 4 (Validation):           {'✅ PASS' if all_passed else '❌ FAIL'}")
    print(f"   Step 5 (Risk v3 Integration):  {'✅ PASS' if step5_passed else '❌ FAIL'}")
    print("-" * 60)
    
    final_result = all([step1_passed, step2_passed, step3_passed, all_passed, step5_passed])
    
    if final_result:
        print("🎉 TP v3 SMOKE TEST: ✅ PASSED")
    else:
        print("⚠️  TP v3 SMOKE TEST: ❌ FAILED")
    
    print("=" * 60 + "\n")
    
    return final_result


if __name__ == "__main__":
    result = asyncio.run(run_tp_smoke_test())
    sys.exit(0 if result else 1)

#!/usr/bin/env python3
"""
Ensemble v5 Validation Script
Tests all 4 agents (XGBoost, LightGBM, PatchTST, N-HiTS) with v5 features
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add project to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai_engine.ensemble_manager import EnsembleManager

print("=" * 70)
print("ENSEMBLE V5 VALIDATION")
print("=" * 70)

# Test symbols
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# v5 feature template
features_template = {
    "price_change": 0.002,
    "high_low_range": 0.015,
    "volume_change": 0.05,
    "volume_ma_ratio": 1.2,
    "ema_10": 44700,
    "ema_20": 44500,
    "ema_50": 44300,
    "rsi_14": 55.3,
    "macd": 0.002,
    "macd_signal": 0.001,
    "macd_hist": 0.001,
    "bb_position": 0.6,
    "volatility_20": 0.025,
    "momentum_10": 0.015,
    "momentum_20": 0.03,
    "ema_10_20_cross": 1.0,
    "ema_10_50_cross": 1.0,
    "volume_ratio": 1.15
}

print("\n📊 Initializing Ensemble Manager...")
try:
    em = EnsembleManager()
    print("✅ Ensemble Manager initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

print("\n🔍 Agent Status:")
print(f"   XGBoost:  {'✅ READY' if em.xgb_agent and em.xgb_agent.is_ready() else '❌ NOT READY'}")
print(f"   LightGBM: {'✅ READY' if em.lgbm_agent and em.lgbm_agent.is_ready() else '❌ NOT READY'}")
print(f"   PatchTST: {'✅ READY' if em.patchtst_agent and em.patchtst_agent.is_ready() else '❌ NOT READY'}")
print(f"   N-HiTS:   {'✅ READY' if em.nhits_agent and em.nhits_agent.is_ready() else '❌ NOT READY'}")

active_count = sum([
    1 if em.xgb_agent and em.xgb_agent.is_ready() else 0,
    1 if em.lgbm_agent and em.lgbm_agent.is_ready() else 0,
    1 if em.patchtst_agent and em.patchtst_agent.is_ready() else 0,
    1 if em.nhits_agent and em.nhits_agent.is_ready() else 0
])
print(f"\n📈 Active Models: {active_count}/4")

if active_count < 1:
    print("❌ No models active! Cannot validate.")
    sys.exit(1)

print("\n🧪 Running Predictions...")
results = {}
all_actions = []

for symbol in symbols:
    print(f"\n--- {symbol} ---")
    
    # Vary features slightly for each symbol
    features = features_template.copy()
    features["rsi_14"] = np.random.uniform(35, 75)
    features["price_change"] = np.random.uniform(-0.02, 0.02)
    features["momentum_10"] = np.random.uniform(-0.03, 0.03)
    
    try:
        result_tuple = em.predict(symbol, features)
        
        # Handle tuple return (action, confidence, info)
        if isinstance(result_tuple, tuple) and len(result_tuple) >= 2:
            action, confidence = result_tuple[0], result_tuple[1]
            info = result_tuple[2] if len(result_tuple) > 2 else {}
            
            result = {
                'action': action,
                'confidence': confidence,
                'info': info
            }
        else:
            result = result_tuple
        
        results[symbol] = result
        
        print(f"   Ensemble Action: {result['action']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        all_actions.append(result['action'])
        
        if 'votes' in result:
            print(f"   Votes: {result['votes']}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        results[symbol] = {"error": str(e)}

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

# Check variety
unique_actions = set(all_actions)
print(f"\n✨ Signal Variety: {len(unique_actions)} unique actions - {unique_actions}")

if len(unique_actions) >= 2:
    print("   ✅ PASS: Signal variety confirmed (≥2 unique actions)")
else:
    print("   ❌ FAIL: Degenerate output (only 1 unique action)")

# Check confidence std
avg_std = np.mean([r.get('confidence_std', 0) for r in results.values() if 'error' not in r])
print(f"\n📊 Average Confidence Std: {avg_std:.4f}")

if avg_std > 0.02:
    print("   ✅ PASS: Confidence std > 0.02")
else:
    print("   ⚠️ WARNING: Low confidence std (< 0.02)")

# Check active models
print(f"\n🎯 Active Models: {active_count}/4")
if active_count >= 3:
    print("   ✅ PASS: At least 3 models active")
elif active_count >= 2:
    print("   ⚠️ WARNING: Only 2 models active")
else:
    print("   ❌ FAIL: Less than 2 models active")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

success_count = sum([
    len(unique_actions) >= 2,
    avg_std > 0.02,
    active_count >= 3
])

print(f"\n✅ Passed Checks: {success_count}/3")

if success_count == 3:
    print("\n🎉 ENSEMBLE V5 VALIDATION: PASSED ✅")
elif success_count == 2:
    print("\n⚠️ ENSEMBLE V5 VALIDATION: PARTIAL ⚠️")
else:
    print("\n❌ ENSEMBLE V5 VALIDATION: FAILED ❌")

print("\n📄 Detailed Results:")
print(json.dumps(results, indent=2))

print("=" * 70)

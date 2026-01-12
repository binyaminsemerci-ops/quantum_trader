#!/usr/bin/env python3
"""
Ensemble v5 Validation Script
Tests all 4 agents: XGBoost, LightGBM, PatchTST, N-HiTS
"""
import sys, json, numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_engine.ensemble_manager import EnsembleManager

print("=" * 70)
print("QUANTUM TRADER - ENSEMBLE V5 VALIDATION")
print("=" * 70)

# Test features (v5 format - 18 features)
test_features = {
    "price_change": 0.015,
    "high_low_range": 0.02,
    "volume_change": 0.1,
    "volume_ma_ratio": 1.15,
    "ema_10": 44700,
    "ema_20": 44500,
    "ema_50": 44200,
    "rsi_14": 55.3,
    "macd": 0.002,
    "macd_signal": 0.001,
    "macd_hist": 0.001,
    "bb_position": 0.65,
    "volatility_20": 0.025,
    "momentum_10": 0.012,
    "momentum_20": 0.018,
    "ema_10_20_cross": 1.0,
    "ema_10_50_cross": 1.0,
    "volume_ratio": 1.2
}

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

print("\n🔧 Initializing Ensemble Manager...")
try:
    em = EnsembleManager(enabled_models=['xgb', 'lgbm', 'nhits', 'patchtst'])
except Exception as e:
    print(f"❌ Error initializing ensemble: {e}")
    sys.exit(1)

# Check agent status
print("\n📊 AGENT STATUS:")
agents = {
    "XGBoost": em.xgb_agent,
    "LightGBM": em.lgbm_agent,
    "PatchTST": em.patchtst_agent,
    "N-HiTS": em.nhits_agent
}

active_count = 0
for name, agent in agents.items():
    if agent and agent.is_ready():
        version = getattr(agent, 'version', 'unknown')
        print(f"   ✅ {name:12s} ACTIVE (version: {version})")
        active_count += 1
    else:
        print(f"   ❌ {name:12s} INACTIVE")

print(f"\n🎯 Active Agents: {active_count}/4")

if active_count < 3:
    print(f"⚠️ WARNING: Less than 3 agents active. Expected 3-4.")
else:
    print(f"✅ PASS: {active_count}/4 agents active")

# Run predictions
print("\n🔮 ENSEMBLE PREDICTIONS:")
print("-" * 70)

results = {}
all_actions = []
all_confidences = []

for symbol in symbols:
    # Vary features slightly for each symbol
    features = test_features.copy()
    features["rsi_14"] = np.random.uniform(35, 75)
    features["price_change"] = np.random.uniform(-0.02, 0.03)
    
    try:
        result = em.predict(symbol, features)
        results[symbol] = result
        
        action = result.get("action", "UNKNOWN")
        confidence = result.get("confidence", 0.0)
        all_actions.append(action)
        all_confidences.append(confidence)
        
        print(f"\n{symbol}:")
        print(f"   Action: {action}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Active Models: {result.get('active_models', [])}")
        
        # Individual agent predictions
        if 'votes' in result:
            for agent_name, vote_info in result['votes'].items():
                agent_action = vote_info.get('action', 'N/A')
                agent_conf = vote_info.get('confidence', 0)
                print(f"      [{agent_name:10s}] {agent_action:4s} (conf={agent_conf:.3f})")
    
    except Exception as e:
        print(f"❌ Error predicting {symbol}: {e}")
        results[symbol] = {"error": str(e)}

# Validation checks
print("\n" + "=" * 70)
print("VALIDATION RESULTS:")
print("=" * 70)

# 1. Active agents check
active_pass = active_count >= 3
print(f"\n1️⃣ Active Agents: {active_count}/4 {'✅ PASS' if active_pass else '❌ FAIL'}")

# 2. Variety check
unique_actions = len(set(all_actions))
variety_pass = unique_actions >= 2
print(f"2️⃣ Prediction Variety: {unique_actions}/3 unique actions {'✅ PASS' if variety_pass else '❌ FAIL'}")
print(f"   Actions: {set(all_actions)}")

# 3. Confidence distribution
if all_confidences:
    conf_mean = np.mean(all_confidences)
    conf_std = np.std(all_confidences)
    conf_pass = conf_std > 0.02
    print(f"3️⃣ Confidence Distribution: mean={conf_mean:.3f}, std={conf_std:.3f} {'✅ PASS' if conf_pass else '❌ FAIL'}")
else:
    conf_pass = False
    print(f"3️⃣ Confidence Distribution: ❌ FAIL (no data)")

# 4. No errors
error_count = sum(1 for r in results.values() if 'error' in r)
error_pass = error_count == 0
print(f"4️⃣ Error Check: {error_count} errors {'✅ PASS' if error_pass else '❌ FAIL'}")

# Overall status
all_pass = active_pass and variety_pass and conf_pass and error_pass
print("\n" + "=" * 70)
if all_pass:
    print("🎉 VALIDATION PASSED: Ensemble v5 is operational!")
else:
    print("⚠️ VALIDATION FAILED: Issues detected (see above)")
print("=" * 70)

# Save report
report = {
    "timestamp": np.datetime64('now').astype(str),
    "active_agents": active_count,
    "agent_status": {name: (agent.is_ready() if agent else False) for name, agent in agents.items()},
    "predictions": results,
    "validation": {
        "active_agents_pass": active_pass,
        "variety_pass": variety_pass,
        "confidence_pass": conf_pass,
        "error_pass": error_pass,
        "overall_pass": all_pass
    },
    "metrics": {
        "unique_actions": unique_actions,
        "confidence_mean": float(conf_mean) if all_confidences else 0,
        "confidence_std": float(conf_std) if all_confidences else 0,
        "error_count": error_count
    }
}

report_file = Path(__file__).parent / "ensemble_v5_validation_report.json"
with open(report_file, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Report saved to: {report_file}")

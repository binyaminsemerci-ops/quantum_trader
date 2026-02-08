#!/usr/bin/env python3
"""
Test script to verify N-HiTS and PatchTST model reconstruction
EXPECTED: Both agents should produce real predictions with variable confidence
"""
import sys
import os
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.agents.unified_agents import NHiTSAgent, PatchTSTAgent
import numpy as np

print("=" * 70)
print("  VERIFYING N-HiTS & PatchTST MODEL RECONSTRUCTION")
print("=" * 70)

# Create diverse test scenarios
test_scenarios = [
    {
        "name": "Bullish Breakout",
        "features": {
            "price_change": 0.025, "rsi_14": 68.5, "macd": 25.3,
            "volume_ratio": 1.8, "momentum_10": 0.035, "high_low_range": 0.032,
            "volume_change": 0.45, "volume_ma_ratio": 1.35, "ema_10": 43200.0,
            "ema_20": 42800.0, "ema_50": 42100.0, "ema_10_20_cross": 0.25,
            "ema_10_50_cross": 0.58, "volatility_20": 0.022, "macd_signal": 18.5,
            "macd_hist": 6.8, "bb_position": 0.85, "momentum_20": 0.042
        }
    },
    {
        "name": "Bearish Breakdown",
        "features": {
            "price_change": -0.032, "rsi_14": 28.3, "macd": -18.7,
            "volume_ratio": 2.1, "momentum_10": -0.028, "high_low_range": 0.038,
            "volume_change": 0.62, "volume_ma_ratio": 1.55, "ema_10": 41500.0,
            "ema_20": 41900.0, "ema_50": 42500.0, "ema_10_20_cross": -0.22,
            "ema_10_50_cross": -0.48, "volatility_20": 0.028, "macd_signal": -12.3,
            "macd_hist": -6.4, "bb_position": 0.15, "momentum_20": -0.035
        }
    },
    {
        "name": "Sideways Neutral",
        "features": {
            "price_change": 0.002, "rsi_14": 48.5, "macd": 1.2,
            "volume_ratio": 0.95, "momentum_10": 0.005, "high_low_range": 0.012,
            "volume_change": -0.08, "volume_ma_ratio": 0.98, "ema_10": 42100.0,
            "ema_20": 42080.0, "ema_50": 42050.0, "ema_10_20_cross": 0.02,
            "ema_10_50_cross": 0.05, "volatility_20": 0.012, "macd_signal": 0.9,
            "macd_hist": 0.3, "bb_position": 0.52, "momentum_20": 0.008
        }
    },
    {
        "name": "High Volatility",
        "features": {
            "price_change": -0.015, "rsi_14": 52.0, "macd": -5.2,
            "volume_ratio": 3.2, "momentum_10": -0.012, "high_low_range": 0.055,
            "volume_change": 0.85, "volume_ma_ratio": 2.15, "ema_10": 42000.0,
            "ema_20": 42100.0, "ema_50": 42200.0, "ema_10_20_cross": -0.08,
            "ema_10_50_cross": -0.18, "volatility_20": 0.048, "macd_signal": -3.8,
            "macd_hist": -1.4, "bb_position": 0.38, "momentum_20": -0.018
        }
    }
]

# Test N-HiTS Agent
print("\n" + "=" * 70)
print("  TEST 1: N-HiTS Agent")
print("=" * 70)
try:
    agent = NHiTSAgent()
    print(f"\n✅ Agent loaded")
    print(f"   Model type: {type(agent.model).__name__}")
    print(f"   PyTorch model: {type(agent.pytorch_model).__name__ if hasattr(agent, 'pytorch_model') and agent.pytorch_model else 'None'}")
    
    results = []
    for scenario in test_scenarios:
        result = agent.predict("BTCUSDT", scenario["features"])
        results.append(result)
        print(f"\n📊 {scenario['name']}:")
        print(f"   Action: {result['action']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    
    # Analyze results
    actions = [r['action'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    print(f"\n📈 Summary:")
    print(f"   Unique actions: {len(set(actions))} (SELL: {actions.count('SELL')}, HOLD: {actions.count('HOLD')}, BUY: {actions.count('BUY')})")
    print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    print(f"   All stuck at 0.5: {'YES ❌' if all(c == 0.5 for c in confidences) else 'NO ✅'}")
    
    if all(c == 0.5 for c in confidences):
        print(f"\n❌ FAILED: Model still using dummy predictions")
    else:
        print(f"\n✅ SUCCESS: Model generating real predictions")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test PatchTST Agent
print("\n" + "=" * 70)
print("  TEST 2: PatchTST Agent")
print("=" * 70)
try:
    agent = PatchTSTAgent()
    print(f"\n✅ Agent loaded")
    print(f"   Model type: {type(agent.model).__name__}")
    print(f"   PyTorch model: {type(agent.pytorch_model).__name__ if hasattr(agent, 'pytorch_model') and agent.pytorch_model else 'None'}")
    
    results = []
    for scenario in test_scenarios:
        result = agent.predict("BTCUSDT", scenario["features"])
        results.append(result)
        print(f"\n📊 {scenario['name']}:")
        print(f"   Action: {result['action']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    
    # Analyze results
    actions = [r['action'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    print(f"\n📈 Summary:")
    print(f"   Unique actions: {len(set(actions))} (SELL: {actions.count('SELL')}, HOLD: {actions.count('HOLD')}, BUY: {actions.count('BUY')})")
    print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    print(f"   All stuck at 0.5: {'YES ❌' if all(c == 0.5 for c in confidences) else 'NO ✅'}")
    
    if all(c == 0.5 for c in confidences):
        print(f"\n❌ FAILED: Model still using dummy predictions")
    else:
        print(f"\n✅ SUCCESS: Model generating real predictions")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("  FINAL VERDICT")
print("=" * 70)
print("""
✅ PASS CRITERIA:
  - PyTorch models reconstructed (not None)
  - Predictions have variable confidence (not all 0.5)
  - Multiple action types predicted across scenarios
  - No errors during prediction
""")

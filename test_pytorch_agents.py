#!/usr/bin/env python3
"""
Test script to reproduce N-HiTS and PatchTST prediction failure
EXPECTED: Both agents fail with "pnl_pred = 0.0" because model is OrderedDict
"""
import sys
import os
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.agents.unified_agents import NHiTSAgent, PatchTSTAgent
import numpy as np

print("=" * 70)
print("  REPRODUCING N-HiTS & PatchTST PREDICTION FAILURE")
print("=" * 70)

# Create test features (18 features matching training data)
test_features = {
    "price_change": 0.015,
    "rsi_14": 55.2,
    "macd": 12.5,
    "volume_ratio": 1.3,
    "momentum_10": 0.018,
    "high_low_range": 0.025,
    "volume_change": 0.12,
    "volume_ma_ratio": 1.05,
    "ema_10": 42500.0,
    "ema_20": 42300.0,
    "ema_50": 41800.0,
    "ema_10_20_cross": 0.15,
    "ema_10_50_cross": 0.45,
    "volatility_20": 0.018,
    "macd_signal": 10.2,
    "macd_hist": 2.3,
    "bb_position": 0.65,
    "momentum_20": 0.022
}

print("\n📊 Test Features:")
print(f"   price_change: {test_features['price_change']:.3f}")
print(f"   rsi_14: {test_features['rsi_14']:.1f}")
print(f"   macd: {test_features['macd']:.1f}")
print(f"   volume_ratio: {test_features['volume_ratio']:.2f}")

# Test N-HiTS Agent
print("\n" + "=" * 70)
print("  TEST 1: N-HiTS Agent")
print("=" * 70)
try:
    agent = NHiTSAgent()
    print(f"\n✅ Agent loaded successfully")
    print(f"   Model type: {type(agent.model)}")
    print(f"   Has .predict(): {hasattr(agent.model, 'predict')}")
    print(f"   Has .forward(): {hasattr(agent.model, 'forward')}")
    
    # Try prediction
    result = agent.predict("BTCUSDT", test_features)
    print(f"\n📈 Prediction Result:")
    print(f"   Action: {result['action']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Version: {result['version']}")
    
    # Check if it's the dummy behavior
    if result['action'] == 'HOLD' and result['confidence'] == 0.5:
        print(f"\n⚠️  WARNING: Got dummy prediction (HOLD, conf=0.5)")
        print(f"   This indicates model.predict() failed in try/except block")
    
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
    print(f"\n✅ Agent loaded successfully")
    print(f"   Model type: {type(agent.model)}")
    print(f"   Has .predict(): {hasattr(agent.model, 'predict')}")
    print(f"   Has .forward(): {hasattr(agent.model, 'forward')}")
    
    # Try prediction
    result = agent.predict("BTCUSDT", test_features)
    print(f"\n📈 Prediction Result:")
    print(f"   Action: {result['action']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Version: {result['version']}")
    
    # Check if it's the dummy behavior
    if result['action'] == 'HOLD' and result['confidence'] == 0.5:
        print(f"\n⚠️  WARNING: Got dummy prediction (HOLD, conf=0.5)")
        print(f"   This indicates model.predict() failed in try/except block")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("  DIAGNOSIS")
print("=" * 70)
print("""
Expected behavior:
  - Both models loaded as OrderedDict (PyTorch state_dict)
  - OrderedDict does NOT have .predict() method
  - try/except catches AttributeError
  - pnl_pred defaults to 0.0
  - Result: HOLD with conf=0.5 (dummy fallback)
  
Solution needed:
  - Reconstruct model architecture from state_dict
  - Load state_dict into initialized nn.Module
  - Implement classification prediction (forward pass + argmax)
""")

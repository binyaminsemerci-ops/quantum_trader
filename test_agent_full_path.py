#!/usr/bin/env python3
"""Test full ensemble prediction path with realistic features"""
import sys
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.agents.unified_agents import XGBoostAgent, LightGBMAgent

# Realistic feature dict (same as AI Engine sends)
features = {
    'price_change': 0.015,
    'rsi_14': 55.2,
    'macd': 12.5,
    'volume_ratio': 1.3,
    'momentum_10': 0.02,
    'high_low_range': 0.03,
    'volume_change': 0.15,
    'volume_ma_ratio': 1.1,
    'ema_10': 95000.0,
    'ema_20': 94500.0,
    'ema_50': 93000.0,
    'ema_10_20_cross': 1.0,
    'ema_10_50_cross': 0.0,
    'volatility_20': 0.025,
    'macd_signal': 10.2,
    'macd_hist': 2.3,
    'bb_position': 0.6,
    'momentum_20': 0.018
}

print("="*70)
print("TESTING FULL AGENT PATH (as used by ensemble_manager)")
print("="*70)

print("\n[1] Initializing XGBoost Agent...")
try:
    xgb_agent = XGBoostAgent()
    print(f"  ✅ Agent ready: {xgb_agent.is_ready()}")
    print(f"  Model: {type(xgb_agent.model).__name__}")
    print(f"  Features expected: {len(xgb_agent.features)}")
    print(f"  Features: {xgb_agent.features[:5]}...")
except Exception as e:
    print(f"  ❌ Init failed: {e}")
    sys.exit(1)

print("\n[2] Calling xgb_agent.predict('BTCUSDT', features)...")
try:
    result = xgb_agent.predict('BTCUSDT', features)
    print(f"  ✅ SUCCESS!")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  ❌ PREDICTION FAILED!")
    print(f"  Error: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n[3] Initializing LightGBM Agent...")
try:
    lgbm_agent = LightGBMAgent()
    print(f"  ✅ Agent ready: {lgbm_agent.is_ready()}")
except Exception as e:
    print(f"  ❌ Init failed: {e}")
    sys.exit(1)

print("\n[4] Calling lgbm_agent.predict('BTCUSDT', features)...")
try:
    result = lgbm_agent.predict('BTCUSDT', features)
    print(f"  ✅ SUCCESS!")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  ❌ PREDICTION FAILED!")
    print(f"  Error: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

import sys
sys.path.insert(0, "/home/qt/quantum_trader")
from ai_engine.agents.xgb_agent import XGBAgent

# 18 features - varied per symbol
features_btc = {  # Bullish signal
    "price_change": 0.02, "rsi_14": 65.0, "macd": 1.5, "volume_ratio": 1.5,
    "momentum_10": 0.03, "high_low_range": 0.04, "volume_change": 0.2,
    "volume_ma_ratio": 1.3, "ema_10": 51000, "ema_20": 50000, "ema_50": 49000,
    "ema_10_20_cross": 1, "ema_10_50_cross": 1, "volatility_20": 0.025,
    "macd_signal": 1.0, "macd_hist": 0.5, "bb_position": 0.75, "momentum_20": 0.025
}

features_eth = {  # Neutral signal
    "price_change": 0.001, "rsi_14": 48.0, "macd": 0.1, "volume_ratio": 1.0,
    "momentum_10": 0.002, "high_low_range": 0.02, "volume_change": 0.05,
    "volume_ma_ratio": 1.0, "ema_10": 3500, "ema_20": 3490, "ema_50": 3480,
    "ema_10_20_cross": 0, "ema_10_50_cross": 0, "volatility_20": 0.015,
    "macd_signal": 0.1, "macd_hist": 0.0, "bb_position": 0.5, "momentum_20": 0.005
}

features_bnb = {  # Bearish signal
    "price_change": -0.03, "rsi_14": 30.0, "macd": -1.2, "volume_ratio": 0.8,
    "momentum_10": -0.02, "high_low_range": 0.05, "volume_change": -0.15,
    "volume_ma_ratio": 0.9, "ema_10": 580, "ema_20": 590, "ema_50": 600,
    "ema_10_20_cross": -1, "ema_10_50_cross": -1, "volatility_20": 0.035,
    "macd_signal": -0.8, "macd_hist": -0.4, "bb_position": 0.1, "momentum_20": -0.02
}

symbol_features = {
    "BTCUSDT": features_btc,
    "ETHUSDT": features_eth,
    "BNBUSDT": features_bnb
}

print("=== V5 TEST (18 features) ===")
try:
    xgb = XGBAgent()
    print(f"XGBoost: OK (expects {xgb.scaler.n_features_in_ if xgb.scaler else 'N/A'} features)")
except Exception as e:
    print(f"XGBoost: FAIL - {e}")
    xgb = None

predictions = []
for sym in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
    features = symbol_features[sym]
    if xgb:
        try:
            action, confidence, model = xgb.predict(sym, features)  # Returns tuple!
            pred = {"action": action, "confidence": confidence, "model": model}
            predictions.append(pred)
            print(f"{sym}: {action} (conf: {confidence:.3f})")
        except Exception as e:
            print(f"{sym}: ERROR - {e}")

if predictions:
    actions = [p["action"] for p in predictions]
    print(f"\nUnique: {set(actions)}")
    confs = [p["confidence"] for p in predictions]
    import numpy as np
    print(f"Confidence: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}")
    
    if len(set(actions)) > 1 or np.std(confs) > 0.02:
        print("✅ VARIETY CONFIRMED!")
    else:
        print("❌ DEGENERACY DETECTED")

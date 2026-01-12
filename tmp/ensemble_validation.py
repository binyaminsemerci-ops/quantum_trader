import sys
sys.path.insert(0, "/home/qt/quantum_trader")
from ai_engine.ensemble_manager import EnsembleManager
import json
import numpy as np

mock_features = {
    "rsi": 55.3, "macd": 0.002, "signal": 0.001,
    "bb_upper": 45000, "bb_middle": 44500, "bb_lower": 44000,
    "volume_sma": 1500000, "atr": 450, "adx": 28.5,
    "cci": 120, "stoch_k": 65, "stoch_d": 62,
    "obv": 5000000, "mfi": 58, "willr": -35,
    "roc": 2.5, "trix": 0.0015, "vwap": 44600,
    "pivot": 44550, "resistance": 45100, "support": 44000,
    "ema_9": 44700, "ema_21": 44400
}

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
em = EnsembleManager()

print("=== AGENT STATUS ===")
xgb_ready = em.xgb_agent and em.xgb_agent.is_ready()
lgbm_ready = em.lgbm_agent is not None
nhits_ready = em.nhits_agent is not None
patchtst_ready = em.patchtst_agent is not None

print(f"XGBoost: {'OK' if xgb_ready else 'FAIL'}")
print(f"LightGBM: {'OK' if lgbm_ready else 'FAIL'}")
print(f"N-HiTS: {'OK' if nhits_ready else 'FAIL'}")
print(f"PatchTST: {'OK' if patchtst_ready else 'FAIL'}")
active_count = sum([xgb_ready, lgbm_ready, nhits_ready, patchtst_ready])
print(f"Active: {active_count}/4")
print()

all_predictions = []
all_confidences = []
all_sources = []

for sym in symbols:
    features = mock_features.copy()
    features["rsi"] = np.random.uniform(40, 70)
    
    try:
        # Returns: (action, confidence, info_dict)
        action, conf, info = em.predict(sym, features)
        source = info.get("models_used", "unknown") if isinstance(info, dict) else str(info)
        
        print(f"{sym}: {action} (conf: {conf:.3f}) - {source}")
        all_predictions.append(action)
        all_confidences.append(conf)
        all_sources.append(source)
    except Exception as e:
        import traceback
        print(f"{sym}: ERROR - {e}")
        print(traceback.format_exc())

print(f"\n=== VARIETY CHECK ===")
unique = set(all_predictions)
print(f"Unique predictions: {unique}")
print(f"Prediction counts: BUY={all_predictions.count('BUY')}, SELL={all_predictions.count('SELL')}, HOLD={all_predictions.count('HOLD')}")

if len(all_confidences) > 0:
    mean_conf = np.mean(all_confidences)
    std_conf = np.std(all_confidences)
    print(f"\nConfidence: mean={mean_conf:.3f}, std={std_conf:.3f}")
    print(f"Degeneracy: {'DETECTED' if len(unique) == 1 or std_conf < 0.02 else 'NONE'}")
else:
    print("No predictions made")

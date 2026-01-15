import sys
sys.path.insert(0, "/home/qt/quantum_trader")
from ai_engine.agents.xgb_agent import XGBAgent
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

print("=== XGBoost Direct Test ===")
agent = XGBAgent()

print(f"Model ready: {agent.is_ready()}")
print(f"Model version: {agent.model_version}")
print(f"Features expected: {len(agent.feature_order) if agent.feature_order else 'dynamic'}")
if agent.feature_order:
    print(f"Feature order: {agent.feature_order[:5]}...")
print()

# Test with 3 different RSI values
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
rsi_values = [35.0, 55.0, 75.0]  # Oversold, neutral, overbought

print("Testing with different RSI values:")
for sym, rsi in zip(symbols, rsi_values):
    features = mock_features.copy()
    features["rsi"] = rsi
    
    try:
        action, conf, source = agent.predict(sym, features)
        print(f"{sym} (RSI={rsi:.1f}): {action} (conf: {conf:.4f}) - {source}")
    except Exception as e:
        print(f"{sym}: ERROR - {e}")
        import traceback
        print(traceback.format_exc())

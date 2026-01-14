#!/usr/bin/env python3
"""
LightGBM v5 Training – aligned with XGBoost v5 (18 features)
Fetches fresh data from Binance, balanced using class_weight
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.clients.binance_market_data_client import BinanceMarketDataClient

# v5 Feature set (aligned with XGBoost v5)
FEATURES_V5 = [
    "price_change", "high_low_range", "volume_change", "volume_ma_ratio",
    "ema_10", "ema_20", "ema_50", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "volatility_20",
    "momentum_10", "momentum_20",
    "ema_10_20_cross", "ema_10_50_cross", "volume_ratio"
]

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LightGBM v5 Training - Fresh from Binance (18 features)")
print("=" * 70)

# Fetch data from Binance
print(f"\n📊 Fetching data from Binance...")
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
bc = BinanceMarketDataClient()
dfs = []

for symbol in symbols:
    print(f"   {symbol}: ", end="", flush=True)
    candles = bc.get_latest_candles(symbol, "1h", limit=1000)
    if candles is not None and len(candles) > 0:
        print(f"{len(candles)} candles")
        dfs.append(candles)
    else:
        print("FAILED")

if not dfs:
    print("❌ ERROR: No data fetched from Binance")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
df_original = df.copy()  # Save original for create_labels
print(f"[INFO] Total raw samples: {len(df)}")

# Calculate v5 features (same as XGBoost v5)
from ops.retrain.fetch_and_train_xgb_v5 import calculate_features, create_labels

print(f"[FEATURES] Calculating 18 v5 features...")
df = calculate_features(df)
df = df.dropna(subset=FEATURES_V5)
print(f"[FEATURES] Calculated {len(FEATURES_V5)} features, {len(df)} valid samples")

# Create labels
print(f"[LABELS] Creating labels...")
df, y = create_labels(df, df_original, threshold=0.015)
df = df.dropna(subset=FEATURES_V5)

# Extract features
X = df[FEATURES_V5].values

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\n[LABELS] Distribution: SELL={counts[0]}, HOLD={counts[1]}, BUY={counts[2]}")

# Train/validation/test split (70/15/15)
print(f"\n[SPLIT] Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)  # 0.176 of 85% ≈ 15% of total
print(f"[SPLIT] Train: {len(X_train)}, Valid: {len(X_val)}, Test: {len(X_test)}")

# Scale features
print(f"\n⚙️ Scaling features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Compute class weights (natural balancing like XGBoost v5)
class_counts = np.bincount(y_train.astype(int))
class_weights = {i: max(class_counts) / c for i, c in enumerate(class_counts)}
print(f"\n⚖️ Class weights: {class_weights}")

# LightGBM parameters (aligned with XGBoost v5 philosophy)
params = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "metric": "multi_logloss",
    "verbosity": 1,
    "random_state": 42
}

# Calculate sample weights for balancing
sample_weights = np.array([class_weights[int(label)] for label in y_train])

print(f"\n[TRAIN] Fitting LightGBM v5 with sample weights...")

# Create datasets with sample weights
train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=sample_weights)
valid_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

# Train
model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1500,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

print(f"[TRAIN] Best iteration: {model.best_iteration}, Best score: {model.best_score['valid_0']['multi_logloss']:.4f}")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best score: {model.best_score}")

# Evaluate
print(f"\n📊 Evaluation on test set:")
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy*100:.2f}%")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["SELL", "HOLD", "BUY"]))

print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"   Rows: Actual | Columns: Predicted")

# Save model, scaler, and metadata
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_file = SAVE_DIR / f"lightgbm_v{timestamp}_v5.pkl"
scaler_file = SAVE_DIR / f"lightgbm_v{timestamp}_v5_scaler.pkl"
meta_file = SAVE_DIR / f"lightgbm_v{timestamp}_v5_meta.json"

print(f"\n💾 Saving model to {model_file}")
with open(model_file, "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"💾 Saving scaler to {scaler_file}")
with open(scaler_file, "wb") as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"💾 Saving metadata to {meta_file}")
metadata = {
    "version": "v5",
    "timestamp": timestamp,
    "features": FEATURES_V5,
    "num_features": len(FEATURES_V5),
    "class_mapping": {0: "SELL", 1: "HOLD", 2: "BUY"},
    "params": params,
    "best_iteration": model.best_iteration,
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "class_distribution": {
        "SELL": int(class_counts[0]),
        "HOLD": int(class_counts[1]),
        "BUY": int(class_counts[2])
    }
}
with open(meta_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n🎉 LightGBM v5 training complete!")
print(f"   Model: {model_file.name}")
print(f"   Scaler: {scaler_file.name}")
print(f"   Meta: {meta_file.name}")
print(f"\n📦 Ready for deployment to production!")
print("=" * 70)

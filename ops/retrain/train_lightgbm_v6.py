#!/usr/bin/env python3
"""
LightGBM v6 Training – 49 features (FEATURES_V6 schema)
Fetches fresh data from Binance, balanced using class_weight

New in v6:
- 49 features (vs 18 in v5)
- Uses calculate_features_v6() from canonical module
- Aligned with ai_engine/common_features.py schema
- Enhanced feature engineering (candlestick patterns, ADX, volume indicators)
"""
import os
import sys
import json
import pickle
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
from ops.retrain.calculate_features_v6 import calculate_features_v6, get_features_v6, create_labels

# v6 Feature set (49 features - canonical schema)
FEATURES_V6 = get_features_v6()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LightGBM v6 Training - Fresh from Binance (49 features)")
print("=" * 70)

# Fetch data from Binance
print(f"\n📊 Fetching data from Binance...")
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT"]
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

# Calculate v6 features (49 features)
print(f"\n[FEATURES] Calculating 49 v6 features...")
df = calculate_features_v6(df)

# Verify features
available_features = [f for f in FEATURES_V6 if f in df.columns]
missing_features = [f for f in FEATURES_V6 if f not in df.columns]

if missing_features:
    print(f"⚠️ WARNING: Missing features: {missing_features}")
    print(f"[FEATURES] Using {len(available_features)}/49 features")
    FEATURES_V6 = available_features
else:
    print(f"[FEATURES] ✅ All 49 features calculated, {len(df)} valid samples")

# Create labels
print(f"\n[LABELS] Creating labels...")
df, y = create_labels(df, df_original, threshold=0.015, lookahead=5)

# Extract features
X = df[FEATURES_V6].values

# Train/validation/test split (70/15/15)
print(f"\n[SPLIT] Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)  # 0.176 of 85% ≈ 15% of total
print(f"[SPLIT] Train: {len(X_train)}, Valid: {len(X_val)}, Test: {len(X_test)}")

# Scale features
print(f"\n⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"[SCALER] Feature dimension: {scaler.n_features_in_}")

# Compute class weights (natural balancing)
class_counts = np.bincount(y_train.astype(int))
class_weights = {i: max(class_counts) / c for i, c in enumerate(class_counts)}
print(f"\n⚖️ Class weights: {class_weights}")

# LightGBM parameters (optimized for 49 features)
params = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,  # Deeper for more features
    "feature_fraction": 0.8,  # Lower to prevent overfitting
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.1,  # L1 regularization
    "lambda_l2": 0.1,  # L2 regularization
    "metric": "multi_logloss",
    "verbosity": 1,
    "random_state": 42
}

# Calculate sample weights for balancing
sample_weights = np.array([class_weights[int(label)] for label in y_train])

print(f"\n[TRAIN] Fitting LightGBM v6 with sample weights...")

# Create datasets with sample weights
train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=sample_weights)
val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=10)
    ]
)

best_iteration = model.best_iteration
print(f"\n[TRAIN] ✅ Training complete (best_iteration={best_iteration})")

# Evaluate on test set
print(f"\n[EVAL] Evaluating on test set...")
y_pred = model.predict(X_test_scaled, num_iteration=best_iteration)
y_pred_labels = np.argmax(y_pred, axis=1)

test_accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\n[METRICS] Test Accuracy: {test_accuracy:.4f}")
print(f"\n[CONFUSION MATRIX]")
print(confusion_matrix(y_test, y_pred_labels))
print(f"\n[CLASSIFICATION REPORT]")
print(classification_report(y_test, y_pred_labels, target_names=['SELL', 'HOLD', 'BUY']))

# Feature importance
feature_importance = model.feature_importance(importance_type='gain')
feature_names = FEATURES_V6
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n[FEATURE IMPORTANCE] Top 15:")
print(importance_df.head(15).to_string(index=False))

# Save model
version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_filename = f"lgbm_v6_{version}.txt"
scaler_filename = f"lgbm_v6_{version}_scaler.pkl"
metadata_filename = f"lgbm_v6_{version}_metadata.json"

model_path = SAVE_DIR / model_filename
scaler_path = SAVE_DIR / scaler_filename
metadata_path = SAVE_DIR / metadata_filename

print(f"\n[SAVE] Saving model...")
model.save_model(str(model_path))
print(f"[SAVE] Model → {model_path}")

# Save scaler
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler → {scaler_path}")

# Save metadata
metadata = {
    "version": f"lgbm_v6_{version}",
    "features": FEATURES_V6,
    "num_features": len(FEATURES_V6),
    "feature_schema": "FEATURES_V6",
    "model_type": "LightGBM",
    "training_date": datetime.utcnow().isoformat() + "Z",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "test_accuracy": float(test_accuracy),
    "best_iteration": int(best_iteration),
    "hyperparameters": params,
    "class_weights": {str(k): float(v) for k, v in class_weights.items()},
    "scaler_n_features": int(scaler.n_features_in_),
    "feature_importance": {
        feature: float(imp) 
        for feature, imp in zip(feature_names, feature_importance)
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Metadata → {metadata_path}")

print(f"\n✅ LightGBM v6 Training Complete!")
print(f"   Model: {model_filename}")
print(f"   Scaler: {scaler_filename}")
print(f"   Metadata: {metadata_filename}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Features: {len(FEATURES_V6)} (v6 schema)")
print(f"\n💡 Next steps:")
print(f"   1. Copy {model_filename} to ai_engine/models/")
print(f"   2. Copy {scaler_filename} to ai_engine/models/")
print(f"   3. Update lgbm_agent.py to load v6 model")
print(f"   4. Test with ensemble_predictor_service")

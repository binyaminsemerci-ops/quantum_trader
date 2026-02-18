#!/usr/bin/env python3
"""
XGBoost v6 Training – 49 features (FEATURES_V6 schema)
Fetches fresh data from Binance, balanced using class_weight

New in v6:
- 49 features (vs 22 in v5 FUTURES)
- Uses calculate_features_v6() from canonical module
- Aligned with ai_engine/common_features.py schema
- SPOT trading focus (no funding rate / open interest)
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
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
print("XGBoost v6 Training - Fresh from Binance (49 features)")
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
sample_weights = np.array([class_weights[int(label)] for label in y_train])
print(f"\n⚖️ Class weights: {class_weights}")

# XGBoost parameters (optimized for 49 features)
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42
}

print(f"\n[TRAIN] Fitting XGBoost v6 with sample weights...")

# Create DMatrix datasets
dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weights)
dval = xgb.DMatrix(X_val_scaled, label=y_val)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Train model
evals = [(dtrain, 'train'), (dval, 'valid')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=10
)

best_iteration = model.best_iteration
print(f"\n[TRAIN] ✅ Training complete (best_iteration={best_iteration})")

# Evaluate on test set
print(f"\n[EVAL] Evaluating on test set...")
y_pred_proba = model.predict(dtest, iteration_range=(0, best_iteration))
y_pred_labels = np.argmax(y_pred_proba, axis=1)

test_accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\n[METRICS] Test Accuracy: {test_accuracy:.4f}")
print(f"\n[CONFUSION MATRIX]")
print(confusion_matrix(y_test, y_pred_labels))
print(f"\n[CLASSIFICATION REPORT]")
print(classification_report(y_test, y_pred_labels, target_names=['SELL', 'HOLD', 'BUY']))

# Feature importance
feature_importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': f'f{i}', 'feature_name': FEATURES_V6[i], 'importance': feature_importance.get(f'f{i}', 0)}
    for i in range(len(FEATURES_V6))
]).sort_values('importance', ascending=False)

print(f"\n[FEATURE IMPORTANCE] Top 15:")
print(importance_df[['feature_name', 'importance']].head(15).to_string(index=False))

# Save model
version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgb_v6_{version}.pkl"
scaler_filename = f"xgb_v6_{version}_scaler.pkl"
features_filename = f"xgb_v6_{version}_features.pkl"
metadata_filename = f"xgb_v6_{version}_metadata.json"

model_path = SAVE_DIR / model_filename
scaler_path = SAVE_DIR / scaler_filename
features_path = SAVE_DIR / features_filename
metadata_path = SAVE_DIR / metadata_filename

print(f"\n[SAVE] Saving model...")

# Save model (XGBoost booster)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"[SAVE] Model → {model_path}")

# Save scaler
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler → {scaler_path}")

# Save feature list (for agent to load exact order)
with open(features_path, 'wb') as f:
    pickle.dump(FEATURES_V6, f)
print(f"[SAVE] Features → {features_path}")

# Save metadata
metadata = {
    "version": f"xgb_v6_{version}",
    "features": FEATURES_V6,
    "num_features": len(FEATURES_V6),
    "feature_schema": "FEATURES_V6",
    "model_type": "XGBoost",
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
        FEATURES_V6[i]: float(feature_importance.get(f'f{i}', 0))
        for i in range(len(FEATURES_V6))
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Metadata → {metadata_path}")

print(f"\n✅ XGBoost v6 Training Complete!")
print(f"   Model: {model_filename}")
print(f"   Scaler: {scaler_filename}")
print(f"   Features: {features_filename}")
print(f"   Metadata: {metadata_filename}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Features: {len(FEATURES_V6)} (v6 schema)")
print(f"\n💡 Next steps:")
print(f"   1. Copy all 4 files to VPS: /home/qt/quantum_trader/ai_engine/models/")
print(f"   2. Remove old xgb_model.pkl and scaler.pkl")
print(f"   3. Symlink or update xgb_agent.py to load v6 model")
print(f"   4. Restart quantum-ensemble-predictor service")
print(f"   5. Monitor logs for '49 features' (not 50 or 22)")

"""
XGBoost v5 – Generalization-Optimized training script
Fixes v4 degeneracy by removing synthetic oversampling and using true class weights.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === 1. LOAD DATA ============================================================
# Try multiple data sources (Docker legacy, ops/retrain, data/)
DATA_PATHS = [
    "ops/retrain/train_full.csv",  # v4 training data
    "data/binance_training_data_full.csv",  # Docker legacy
    "data/binance_training_data.csv",
]

df = None
for path in DATA_PATHS:
    if os.path.exists(path):
        print(f"[INFO] Found data at {path}")
        df = pd.read_csv(path)
        print(f"[INFO] Dataset shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)[:5]}...")
        break

assert df is not None, f"No training data found! Tried: {DATA_PATHS}"

# Check if we have features or need to extract them
if "label" in df.columns:
    # v4 format: features + label
    X = df.drop(columns=["label"])
    y = df["label"]
    print(f"[INFO] Using v4 format with 'label' column")
elif "target" in df.columns:
    # Standard format: features + target
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"[INFO] Using standard format with 'target' column")
else:
    raise ValueError(f"Cannot find target column ('label' or 'target') in dataset. Columns: {df.columns.tolist()}")

print(f"[INFO] Features: {len(X.columns)}")
print(f"[INFO] Target distribution: {np.bincount(y.astype(int))}")

# === 2. SPLIT DATA ===========================================================
# 70% train, 20% validation, 10% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp
)

print(f"[INFO] Train samples: {len(X_train)}")
print(f"[INFO] Valid samples: {len(X_valid)}")
print(f"[INFO] Test samples: {len(X_test)}")

# === 3. SCALER ===============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# === 4. CLASS WEIGHTS (no oversampling) =====================================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
sample_weights = np.array([class_weight_dict[cls] for cls in y_train])

print(f"\n[INFO] Class distribution (train):")
for cls in classes:
    count = np.sum(y_train == cls)
    pct = 100 * count / len(y_train)
    print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

print(f"\n[INFO] Computed class weights: {class_weight_dict}")
print(f"[INFO] Using sample_weight during training (no synthetic data)")

# === 5. TRAIN MODEL ==========================================================
params = dict(
    objective="multi:softprob",
    num_class=3,
    learning_rate=0.025,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.2,
    reg_lambda=2.0,
    reg_alpha=0.4,
    n_estimators=800,
    eval_metric="mlogloss",
    verbosity=1,
    random_state=42,
)

xgb = XGBClassifier(**params)

print("\n[TRAIN] Fitting XGBoost v5 model with class weights...")
print(f"[TRAIN] Parameters: {params}")

xgb.fit(
    X_train_scaled,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_valid_scaled, y_valid)],
    early_stopping_rounds=40,
    verbose=True,
)

print(f"\n[TRAIN] Best iteration: {xgb.best_iteration}")
print(f"[TRAIN] Best score: {xgb.best_score:.4f}")

# === 6. EVALUATE =============================================================
print("\n=== VALIDATION SET EVALUATION ===")
y_valid_pred = np.argmax(xgb.predict_proba(X_valid_scaled), axis=1)
valid_report = classification_report(y_valid, y_valid_pred, digits=4)
valid_matrix = confusion_matrix(y_valid, y_valid_pred)
print(valid_report)
print("Confusion Matrix (Validation):")
print(valid_matrix)

print("\n=== TEST SET EVALUATION ===")
y_test_pred = np.argmax(xgb.predict_proba(X_test_scaled), axis=1)
test_report = classification_report(y_test, y_test_pred, digits=4)
test_matrix = confusion_matrix(y_test, y_test_pred)
print(test_report)
print("Confusion Matrix (Test):")
print(test_matrix)

# Check for degeneracy
test_pred_unique = np.unique(y_test_pred)
print(f"\n[CHECK] Unique predictions in test set: {test_pred_unique}")
if len(test_pred_unique) < 3:
    print(f"[WARNING] Model is degenerate! Only predicting classes: {test_pred_unique}")
else:
    print(f"[SUCCESS] Model predicts all 3 classes ✓")

# === 7. SAVE ARTIFACTS ======================================================
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_name = f"xgb_v{timestamp}_v5.pkl"
scaler_name = f"xgb_v{timestamp}_v5_scaler.pkl"
meta_name = f"xgb_v{timestamp}_v5_meta.json"

model_dir = "ai_engine/models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, model_name)
scaler_path = os.path.join(model_dir, scaler_name)
meta_path = os.path.join(model_dir, meta_name)

joblib.dump(xgb, model_path)
joblib.dump(scaler, scaler_path)

meta = {
    "version": "v5",
    "timestamp": timestamp,
    "features": list(X.columns),
    "class_mapping": {0: "SELL", 1: "HOLD", 2: "BUY"},
    "train_samples": len(X_train),
    "valid_samples": len(X_valid),
    "test_samples": len(X_test),
    "params": params,
    "class_weights": {int(k): float(v) for k, v in class_weight_dict.items()},
    "best_iteration": int(xgb.best_iteration),
    "best_score": float(xgb.best_score),
    "test_report": test_report,
    "test_confusion_matrix": test_matrix.tolist(),
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Model saved: {model_path}")
print(f"✅ Scaler saved: {scaler_path}")
print(f"✅ Meta saved: {meta_path}")

# === 8. VALIDATION TEST SAMPLE ==============================================
print("\n=== SAMPLE OUTPUT CHECK (first 5 test samples) ===")
sample = X_test_scaled[:5]
proba = xgb.predict_proba(sample)
pred = np.argmax(proba, axis=1)

for i, p in enumerate(pred):
    class_name = ["SELL", "HOLD", "BUY"][p]
    print(f"Sample {i}: {class_name} (class={p}) probs={np.round(proba[i], 3)}")

# Check confidence variance
confidences = np.max(proba, axis=1)
conf_mean = np.mean(confidences)
conf_std = np.std(confidences)
print(f"\n[CONFIDENCE] Mean: {conf_mean:.4f}, Std: {conf_std:.4f}")

if conf_std < 0.02:
    print(f"[WARNING] Low confidence variance ({conf_std:.4f}) - possible degeneracy!")
else:
    print(f"[SUCCESS] Confidence variance OK ({conf_std:.4f} >= 0.02)")

print("\n" + "="*60)
print("XGBoost v5 training complete!")
print("Next steps:")
print("1. Restart quantum-ai-engine service")
print("2. Run validation: sudo -u qt /opt/quantum/venvs/ai-engine/bin/python3 /tmp/ensemble_validation.py")
print("="*60)

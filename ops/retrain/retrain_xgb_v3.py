import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

# === Paths & setup ===
DATA_PATH = "ops/retrain/train_full.csv"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = f"xgb_v{datetime.now().strftime('%Y%m%d_%H%M%S')}_v3"
MODEL_PATH = OUTPUT_DIR / f"{MODEL_NAME}.pkl"
SCALER_PATH = OUTPUT_DIR / f"{MODEL_NAME}_scaler.pkl"
META_PATH = OUTPUT_DIR / f"{MODEL_NAME}_meta.json"

# === Load data ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).astype(np.float32)
y = df["label"].astype(np.int32)
features = list(X.columns)

print(f"📊 Training XGBoost v3 with {len(features)} features, {len(X)} samples")

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"🧩 Scaler saved to {SCALER_PATH}")

# === Split data ===
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Compute class balance ===
counts = np.bincount(y)
scale_pos_weight = float(counts[0] / max(1, counts[2])) if len(counts) >= 3 else 1.0
print(f"⚖️ Class distribution: {counts.tolist()} | scale_pos_weight={scale_pos_weight:.3f}")

# === Build DMatrix ===
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

# === Model params ===
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "eta": 0.03,
    "max_depth": 9,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "lambda": 1.5,
    "alpha": 0.1,
    "scale_pos_weight": scale_pos_weight,
    "tree_method": "hist",
    "nthread": 4,
}

# === Train model ===
print("🚀 Training...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1500,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=100
)
print("✅ Training complete.")

# === Save model & metadata ===
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

metadata = {
    "timestamp": datetime.now().isoformat(),
    "num_features": len(features),
    "features": features,
    "num_class": 3,
    "params": params,
    "best_iteration": int(model.best_iteration),
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"🧠 Metadata saved to {META_PATH}")

# === Test prediction ===
test_pred = model.predict(xgb.DMatrix(X_val[:5], feature_names=features))
probs = np.round(test_pred.mean(axis=0), 4)
print(f"🔍 Avg output distribution: {probs} (sum={probs.sum():.2f})")

print("\n✅ XGBoost v3 training finished successfully.")

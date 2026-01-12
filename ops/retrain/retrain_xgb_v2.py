import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import joblib
import json

# === Konfigurasjon ===
DATA_PATH = "ops/retrain/train_full.csv"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = f"xgb_v{datetime.now().strftime('%Y%m%d_%H%M%S')}_v2"
MODEL_PATH = OUTPUT_DIR / f"{MODEL_NAME}.pkl"
METADATA_PATH = OUTPUT_DIR / f"{MODEL_NAME}_meta.json"

# === Last treningsdata ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).astype(np.float32)
y = df["label"].astype(np.int32)
features = list(X.columns)

print(f"📊 Training XGBoost v2 with {len(features)} features, {len(X)} samples")

# === Konfigurer XGBoost model ===
params = {
    "objective": "multi:softprob",
    "num_class": 3,  # SELL / HOLD / BUY
    "eval_metric": "mlogloss",
    "eta": 0.05,
    "max_depth": 7,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "lambda": 1.5,
    "alpha": 0.1,
    "tree_method": "hist",
    "nthread": 4,
}

dtrain = xgb.DMatrix(X, label=y, feature_names=features)

# === Tren model ===
print("🚀 Training...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    verbose_eval=50
)
print("✅ Training complete")

# === Lagre modell ===
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")

# === Lagre metadata ===
metadata = {
    "timestamp": datetime.now().isoformat(),
    "num_features": len(features),
    "features": features,
    "num_class": 3,
    "params": params
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"🧩 Metadata saved: {METADATA_PATH}")

# === Valider output ===
test_sample = X.iloc[0:1]
pred = model.predict(xgb.DMatrix(test_sample))
pred_label = int(np.argmax(pred))
print(f"🔍 Test prediction: {pred_label} (probabilities = {np.round(pred, 4).tolist()})")

print("\n✅ XGBoost v2 training pipeline complete.")
print("   Use model in production as: models/xgb_v*_v2.pkl")

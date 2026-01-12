import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb

# === CONFIG ===
MODEL_VERSION = "v4"
MODEL_DIR = "models"
SEED = 42
os.makedirs(MODEL_DIR, exist_ok=True)

def log(msg: str):
    print(f"[XGB_v4] {msg}")

# === LOAD DATA ===
DATA_PATH = "ops/retrain/train_full.csv"
df = pd.read_csv(DATA_PATH)
log(f"Loaded dataset: {df.shape}")

# === BASIC PREP ===
X = df.drop(columns=["label"])
y = df["label"]

log(f"Class distribution before balancing:\n{y.value_counts(normalize=True)}")

# === STRATIFIED SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)

# === BALANCE CLASSES ===
ros = RandomOverSampler(random_state=SEED)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
log(f"Balanced class distribution:\n{pd.Series(y_train_res).value_counts(normalize=True)}")

# === SCALING ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)

# === DMatrix ===
dtrain = xgb.DMatrix(X_train_scaled, label=y_train_res)
dval = xgb.DMatrix(X_val_scaled, label=y_val)

# === MODEL PARAMS ===
params = {
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "lambda": 1.0,
    "alpha": 0.2,
    "seed": SEED,
    "tree_method": "hist",
    "nthread": 8,
}

# === TRAIN ===
log("Starting training with stratified oversampling...")
evals = [(dtrain, "train"), (dval, "val")]

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1500,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50,
)

# === EVALUATION ===
y_pred_prob = booster.predict(dval)
y_pred = np.argmax(y_pred_prob, axis=1)
log("Validation results:")
log(classification_report(y_val, y_pred))
log(f"Confusion matrix:\n{confusion_matrix(y_val, y_pred)}")

# === SAVE MODEL + SCALER + META ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"xgb_v{timestamp}_{MODEL_VERSION}"
model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")
meta_path = os.path.join(MODEL_DIR, f"{model_name}_meta.json")

joblib.dump(booster, model_path)
joblib.dump(scaler, scaler_path)

meta = {
    "version": MODEL_VERSION,
    "timestamp": timestamp,
    "features": list(X.columns),
    "num_features": len(X.columns),
    "num_classes": len(np.unique(y)),
    "train_shape": X_train_res.shape,
    "val_shape": X_val.shape,
    "balanced_distribution": pd.Series(y_train_res).value_counts().to_dict(),
    "scaler": scaler_path,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

log(f"✅ Model saved: {model_path}")
log(f"✅ Scaler saved: {scaler_path}")
log(f"✅ Metadata saved: {meta_path}")
log(f"Best iteration: {booster.best_iteration}")
log("Training complete.")

# === Optional: Summary CSV ===
summary_path = os.path.join(MODEL_DIR, f"{model_name}_summary.csv")
report_df = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).T
report_df.to_csv(summary_path)
log(f"Saved evaluation report: {summary_path}")

print("\n=== TRAINING DONE ===")
print(f"Model: {model_path}")
print(f"Scaler: {scaler_path}")
print(f"Meta: {meta_path}")

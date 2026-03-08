#!/usr/bin/env python3
"""XGBoost v6 - 49 features (FEATURES_V6 schema) - fresh Binance data"""
import os, sys, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.clients.binance_market_data_client import BinanceMarketDataClient
from ops.retrain.calculate_features_v6 import calculate_features_v6, get_features_v6, create_labels

FEATURES_V6 = get_features_v6()
BASE_DIR = Path(__file__).parent.parent.parent
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("XGBoost v6 Training - 49 features")
print("=" * 60)

bc = BinanceMarketDataClient()
symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT"]
dfs = []
for sym in symbols:
    candles = bc.get_latest_candles(sym, "1h", limit=1000)
    if candles is not None and len(candles) > 0:
        print(f"  {sym}: {len(candles)} candles")
        dfs.append(candles)

if not dfs:
    print("ERROR: no data"); sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
df_orig = df.copy()
df = calculate_features_v6(df)
avail = [f for f in FEATURES_V6 if f in df.columns]
print(f"Features available: {len(avail)}/49")
FEATURES_V6 = avail

df, y = create_labels(df, df_orig, threshold=0.015, lookahead=5)
X = df[FEATURES_V6].values
print(f"Samples with labels: {len(X)}, Features: {len(FEATURES_V6)}")

X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.176, random_state=42, stratify=y_t)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", tree_method="hist",
    n_jobs=4, random_state=42
)
model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=50)
y_pred = model.predict(X_test_s)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["SELL","HOLD","BUY"]))

version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
name = f"xgboost_v{version}_v6"
model_path = SAVE_DIR / f"{name}.pkl"
scaler_path = SAVE_DIR / f"{name}_scaler.pkl"
meta_path   = SAVE_DIR / f"{name}_meta.json"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
with open(meta_path, "w") as f:
    json.dump({"version": f"xgb_v6_{version}", "features": FEATURES_V6}, f)

print(f"Model  -> {model_path}")
print(f"Scaler -> {scaler_path}")
print(f"Meta   -> {meta_path} ({len(FEATURES_V6)} features)")
print("DONE")

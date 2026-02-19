"""
Fetch live data from Binance Testnet, extract features, and train XGBoost v5
"""

import os
import sys
import json
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === 1. FETCH DATA FROM BINANCE MAINNET (PUBLIC) ============================
def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    """Fetch historical klines from Binance mainnet (public API, no key needed)"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)  # Max 1000 per request
    }
    
    print(f"[FETCH] Getting {params['limit']} {interval} candles for {symbol}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to float and timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    print(f"[FETCH] Got {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# === 2. CALCULATE TECHNICAL FEATURES =========================================
def calculate_features(df):
    """Calculate 23 technical features matching v4 metadata"""
    print("[FEATURES] Calculating technical indicators...")
    
    df = df.copy()
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # EMAs
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # EMA crosses
    df['ema_10_20_cross'] = (df['ema_10'] > df['ema_20']).astype(int)
    df['ema_10_50_cross'] = (df['ema_10'] > df['ema_50']).astype(int)
    
    # Bollinger Bands position
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Volatility
    df['volatility_20'] = df['close'].rolling(20).std() / df['close']
    
    # 18 technical features (NO OHLCV - production only sends these!)
    feature_cols = [
        'price_change', 'rsi_14', 'macd', 'volume_ratio', 'momentum_10',
        'high_low_range', 'volume_change', 'volume_ma_ratio',
        'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 'ema_10_50_cross',
        'volatility_20', 'macd_signal', 'macd_hist', 'bb_position', 'momentum_20'
    ]
    
    # Drop NaN rows (from rolling calculations)
    df = df.dropna()
    
    # Replace inf with NaN and drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    print(f"[FEATURES] Calculated {len(feature_cols)} features, {len(df)} valid samples")
    return df[feature_cols]

# === 3. CREATE LABELS ========================================================
def create_labels(df_with_features, df_original, threshold=0.015):
    """
    Create labels based on future price movement
    0 = SELL (price drops > threshold)
    1 = HOLD (price changes < threshold)
    2 = BUY (price rises > threshold)
    """
    print(f"[LABELS] Creating labels with threshold={threshold*100}%...")
    
    # Align indices
    df_original = df_original.loc[df_with_features.index]
    
    # Calculate future return (looking 5 periods ahead)
    future_return = df_original['close'].shift(-5) / df_original['close'] - 1
    
    labels = np.where(future_return > threshold, 2,  # BUY
                     np.where(future_return < -threshold, 0,  # SELL
                             1))  # HOLD
    
    # Drop last 5 rows (no future data)
    df_with_features = df_with_features.iloc[:-5]
    labels = labels[:-5]
    
    label_counts = np.bincount(labels)
    print(f"[LABELS] Distribution: SELL={label_counts[0]}, HOLD={label_counts[1]}, BUY={label_counts[2]}")
    
    return df_with_features, labels

# === MAIN TRAINING PIPELINE ==================================================
print("="*60)
print("XGBoost v5 Training - Live Data from Binance Mainnet")
print("="*60)

# Fetch data for multiple symbols (1000 candles each = ~42 days at 1h)
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
all_data = []

for symbol in symbols:
    try:
        df = fetch_binance_data(symbol=symbol, interval="1h", limit=1000)
        df['symbol'] = symbol
        all_data.append(df)
    except Exception as e:
        print(f"[WARNING] Failed to fetch {symbol}: {e}")

if not all_data:
    raise ValueError("No data fetched!")

df_raw = pd.concat(all_data, ignore_index=True)
print(f"\n[INFO] Total raw samples: {len(df_raw)}")

# Calculate features
df_features = calculate_features(df_raw)

# Create labels
X, y = create_labels(df_features, df_raw)

print(f"\n[INFO] Final dataset: {len(X)} samples, {len(X.columns)} features")

# === SPLIT DATA ==============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp
)

print(f"\n[SPLIT] Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

# === SCALER ==================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# === CLASS WEIGHTS ===========================================================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
sample_weights = np.array([class_weight_dict[cls] for cls in y_train])

print(f"\n[WEIGHTS] Class distribution (train):")
for cls in classes:
    count = np.sum(y_train == cls)
    pct = 100 * count / len(y_train)
    print(f"  Class {cls}: {count} samples ({pct:.1f}%)")
print(f"[WEIGHTS] Computed class weights: {class_weight_dict}")

# === TRAIN ===================================================================
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
    early_stopping_rounds=40,
    verbosity=1,
    random_state=42,
)

xgb = XGBClassifier(**params)

print("\n[TRAIN] Fitting XGBoost v5 with class weights...")
xgb.fit(
    X_train_scaled,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_valid_scaled, y_valid)],
    verbose=True,
)

print(f"\n[TRAIN] Best iteration: {xgb.best_iteration}")
print(f"[TRAIN] Best score: {xgb.best_score:.4f}")

# === EVALUATE ================================================================
print("\n=== VALIDATION SET ===")
y_valid_pred = np.argmax(xgb.predict_proba(X_valid_scaled), axis=1)
print(classification_report(y_valid, y_valid_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred))

print("\n=== TEST SET ===")
y_test_pred = np.argmax(xgb.predict_proba(X_test_scaled), axis=1)
test_report = classification_report(y_test, y_test_pred, digits=4)
test_matrix = confusion_matrix(y_test, y_test_pred)
print(test_report)
print("Confusion Matrix:")
print(test_matrix)

# Degeneracy check
test_pred_unique = np.unique(y_test_pred)
print(f"\n[CHECK] Unique predictions: {test_pred_unique}")
if len(test_pred_unique) < 3:
    print(f"[WARNING] Degeneracy detected! Only predicting: {test_pred_unique}")
else:
    print(f"[SUCCESS] All 3 classes predicted ✓")

# === SAVE ====================================================================
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_name = f"xgb_v{timestamp}_v5.pkl"
scaler_name = f"xgb_v{timestamp}_v5_scaler.pkl"
meta_name = f"xgb_v{timestamp}_v5_meta.json"

model_dir = "ai_engine/models"
os.makedirs(model_dir, exist_ok=True)

# Use pickle.dump() instead of joblib for agent compatibility
import pickle
with open(os.path.join(model_dir, model_name), 'wb') as f:
    pickle.dump(xgb, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(model_dir, scaler_name), 'wb') as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

meta = {
    "version": "v5",
    "timestamp": timestamp,
    "features": list(X.columns),
    "class_mapping": {0: "SELL", 1: "HOLD", 2: "BUY"},
    "train_samples": len(X_train),
    "valid_samples": len(X_valid),
    "test_samples": len(X_test),
    "symbols": symbols,
    "params": params,
    "class_weights": {int(k): float(v) for k, v in class_weight_dict.items()},
    "best_iteration": int(xgb.best_iteration),
    "best_score": float(xgb.best_score),
    "test_report": test_report,
}

with open(os.path.join(model_dir, meta_name), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Model: {os.path.join(model_dir, model_name)}")
print(f"✅ Scaler: {os.path.join(model_dir, scaler_name)}")
print(f"✅ Meta: {os.path.join(model_dir, meta_name)}")

# === SAMPLE CHECK ============================================================
print("\n=== SAMPLE PREDICTIONS ===")
sample = X_test_scaled[:5]
proba = xgb.predict_proba(sample)
pred = np.argmax(proba, axis=1)

for i, p in enumerate(pred):
    class_name = ["SELL", "HOLD", "BUY"][p]
    print(f"Sample {i}: {class_name} probs={np.round(proba[i], 3)}")

confidences = np.max(proba, axis=1)
conf_std = np.std(confidences)
print(f"\n[CONFIDENCE] Mean: {np.mean(confidences):.4f}, Std: {conf_std:.4f}")

if conf_std < 0.02:
    print(f"[WARNING] Low variance - possible degeneracy!")
else:
    print(f"[SUCCESS] Good confidence variance ✓")

print("\n" + "="*60)
print("Training complete! Restart service and run validation.")
print("="*60)

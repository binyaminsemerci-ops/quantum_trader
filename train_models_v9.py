#!/usr/bin/env python3
"""
Train NHiTS and PatchTST models v9.
Import wrapper from ai_engine.agents.model_wrappers to ensure pickle compatibility.
RUN THIS ON VPS: cd /opt/quantum && python3 train_models_v9.py
"""
import sys
sys.path.insert(0, '/opt/quantum')

import os, json, requests, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import joblib

# CRITICAL: Import from the module that will be available on AI Engine
from ai_engine.agents.model_wrappers import SimpleMLP, TorchRegressorWrapper

FEATURES_V5 = [
    "price_change", "high_low_range", "volume_change", "volume_ma_ratio",
    "ema_10", "ema_20", "ema_50", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "volatility_20",
    "momentum_10", "momentum_20",
    "ema_10_20_cross", "ema_10_50_cross", "volume_ratio"
]

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "ARBUSDT"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("/home/qt/quantum_trader/models")
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
PATIENCE = 10

print("=" * 70)
print("NHiTS + PatchTST v9 Training")
print(f"Device: {DEVICE}")
print("=" * 70)

def fetch_binance_klines(symbol, interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
    except Exception as e:
        print(f"   Error: {e}")
    return pd.DataFrame()

def fetch_all_data():
    print("\n📊 Fetching data from Binance...")
    all_data = []
    for symbol in SYMBOLS:
        print(f"   {symbol}...", end=" ")
        df = fetch_binance_klines(symbol, limit=1000)
        if not df.empty:
            print(f"{len(df)} rows")
            all_data.append(df)
        else:
            print("FAILED")
        time.sleep(0.2)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total: {len(combined)} rows from {len(all_data)} symbols")
        return combined
    return pd.DataFrame()

def calculate_features(df):
    df = df.copy().sort_values(["symbol", "timestamp"])
    features_list = []
    
    for symbol in df["symbol"].unique():
        sdf = df[df["symbol"] == symbol].copy()
        if len(sdf) < 60:
            continue
        
        sdf["price_change"] = sdf["close"].pct_change()
        sdf["high_low_range"] = (sdf["high"] - sdf["low"]) / sdf["close"]
        sdf["volume_change"] = sdf["volume"].pct_change()
        sdf["volume_ma"] = sdf["volume"].rolling(20).mean()
        sdf["volume_ma_ratio"] = sdf["volume"] / sdf["volume_ma"]
        sdf["volume_ratio"] = sdf["volume"] / sdf["volume"].shift(1)
        
        sdf["ema_10"] = sdf["close"].ewm(span=10).mean() / sdf["close"] - 1
        sdf["ema_20"] = sdf["close"].ewm(span=20).mean() / sdf["close"] - 1
        sdf["ema_50"] = sdf["close"].ewm(span=50).mean() / sdf["close"] - 1
        
        delta = sdf["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        sdf["rsi_14"] = (100 - (100 / (1 + rs))) / 100
        
        ema12 = sdf["close"].ewm(span=12).mean()
        ema26 = sdf["close"].ewm(span=26).mean()
        sdf["macd"] = (ema12 - ema26) / sdf["close"]
        sdf["macd_signal"] = sdf["macd"].ewm(span=9).mean()
        sdf["macd_hist"] = sdf["macd"] - sdf["macd_signal"]
        
        bb_mid = sdf["close"].rolling(20).mean()
        bb_std = sdf["close"].rolling(20).std()
        sdf["bb_position"] = (sdf["close"] - bb_mid) / (2 * bb_std + 1e-10)
        
        sdf["volatility_20"] = sdf["close"].pct_change().rolling(20).std()
        sdf["momentum_10"] = sdf["close"].pct_change(10)
        sdf["momentum_20"] = sdf["close"].pct_change(20)
        
        ema10_raw = sdf["close"].ewm(span=10).mean()
        ema20_raw = sdf["close"].ewm(span=20).mean()
        ema50_raw = sdf["close"].ewm(span=50).mean()
        sdf["ema_10_20_cross"] = (ema10_raw - ema20_raw) / sdf["close"]
        sdf["ema_10_50_cross"] = (ema10_raw - ema50_raw) / sdf["close"]
        
        # Regression target: future PnL%
        sdf["pnl_target"] = (sdf["close"].shift(-5) / sdf["close"] - 1) * 100
        
        features_list.append(sdf)
    
    if features_list:
        result = pd.concat(features_list)
        result = result.dropna().replace([np.inf, -np.inf], 0)
        return result
    return pd.DataFrame()

class TradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    print(f"\n🚀 Training {model_name}...")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"   ✅ Best val loss: {best_val_loss:.4f}")
    return model, best_val_loss


if __name__ == "__main__":
    df = fetch_all_data()
    if df.empty:
        print("❌ No data!")
        sys.exit(1)
    
    print("\n⚙️  Calculating features...")
    df_features = calculate_features(df)
    print(f"   {len(df_features)} samples")
    
    X = df_features[FEATURES_V5].values
    y = df_features["pnl_target"].values
    y = np.clip(y, -10, 10)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)
    X_scaled = np.nan_to_num(X_scaled, 0)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"\n📈 train={len(X_train)}, val={len(X_val)}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train NHiTS
    print("\n" + "=" * 70)
    nhits = SimpleMLP(input_size=18, hidden_size=128, output_size=1)
    nhits, nhits_loss = train_model(nhits, train_loader, val_loader, "NHiTS-v9")
    nhits.eval()
    
    # Save using TorchRegressorWrapper from ai_engine.agents.model_wrappers
    nhits_state = {k: v.cpu() for k, v in nhits.state_dict().items()}
    nhits_wrapper = TorchRegressorWrapper(nhits_state, {'input_size': 18, 'hidden_size': 128})
    
    nhits_path = SAVE_DIR / f"nhits_v{timestamp}_v9.pkl"
    joblib.dump(nhits_wrapper, nhits_path)
    print(f"💾 Saved: {nhits_path}")
    
    joblib.dump(scaler, SAVE_DIR / f"nhits_v{timestamp}_v9_scaler.pkl")
    with open(SAVE_DIR / f"nhits_v{timestamp}_v9_meta.json", 'w') as f:
        json.dump({"features": FEATURES_V5, "version": f"nhits_v9_{timestamp}"}, f)
    
    # Train PatchTST
    print("\n" + "=" * 70)
    patchtst = SimpleMLP(input_size=18, hidden_size=64, output_size=1)
    patchtst, patchtst_loss = train_model(patchtst, train_loader, val_loader, "PatchTST-v9")
    patchtst.eval()
    
    patchtst_state = {k: v.cpu() for k, v in patchtst.state_dict().items()}
    patchtst_wrapper = TorchRegressorWrapper(patchtst_state, {'input_size': 18, 'hidden_size': 64})
    
    patchtst_path = SAVE_DIR / f"patchtst_v{timestamp}_v9.pkl"
    joblib.dump(patchtst_wrapper, patchtst_path)
    print(f"💾 Saved: {patchtst_path}")
    
    joblib.dump(scaler, SAVE_DIR / f"patchtst_v{timestamp}_v9_scaler.pkl")
    with open(SAVE_DIR / f"patchtst_v{timestamp}_v9_meta.json", 'w') as f:
        json.dump({"features": FEATURES_V5, "version": f"patchtst_v9_{timestamp}"}, f)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print(f"   NHiTS loss: {nhits_loss:.4f}")
    print(f"   PatchTST loss: {patchtst_loss:.4f}")
    print("=" * 70)

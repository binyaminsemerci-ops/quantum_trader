#!/usr/bin/env python3
"""
NHiTS + PatchTST v6 Training
Fetches fresh data from Binance and trains models with 18 features
"""
import os, sys, json, pickle, requests, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader

# === Configuration ===
FEATURES_V5 = [
    "price_change", "high_low_range", "volume_change", "volume_ma_ratio",
    "ema_10", "ema_20", "ema_50", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "volatility_20",
    "momentum_10", "momentum_20",
    "ema_10_20_cross", "ema_10_50_cross", "volume_ratio"
]

# Priority symbols for training
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "ARBUSDT"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("/opt/quantum/models")
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
PATIENCE = 10

print("=" * 70)
print("NHiTS + PatchTST v6 Training")
print(f"Device: {DEVICE}")
print(f"Symbols: {len(SYMBOLS)}")
print("=" * 70)

# === Binance Data Fetching ===
def fetch_binance_klines(symbol, interval="1h", limit=1000):
    """Fetch historical klines from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
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
        print(f"   Error fetching {symbol}: {e}")
    return pd.DataFrame()

def fetch_all_data():
    """Fetch data for all symbols"""
    print("\n📊 Fetching data from Binance...")
    all_data = []
    for symbol in SYMBOLS:
        print(f"   Fetching {symbol}...", end=" ")
        df = fetch_binance_klines(symbol, limit=1000)
        if not df.empty:
            print(f"{len(df)} candles")
            all_data.append(df)
        else:
            print("FAILED")
        time.sleep(0.2)  # Rate limit
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total: {len(combined)} rows from {len(all_data)} symbols")
        return combined
    return pd.DataFrame()

# === Feature Engineering ===
def calculate_features(df):
    """Calculate all 18 v5 features"""
    df = df.copy().sort_values(["symbol", "timestamp"])
    features_list = []
    
    for symbol in df["symbol"].unique():
        sdf = df[df["symbol"] == symbol].copy()
        if len(sdf) < 60:
            continue
        
        # Price features
        sdf["price_change"] = sdf["close"].pct_change()
        sdf["high_low_range"] = (sdf["high"] - sdf["low"]) / sdf["close"]
        
        # Volume features
        sdf["volume_change"] = sdf["volume"].pct_change()
        sdf["volume_ma"] = sdf["volume"].rolling(20).mean()
        sdf["volume_ma_ratio"] = sdf["volume"] / sdf["volume_ma"]
        sdf["volume_ratio"] = sdf["volume"] / sdf["volume"].shift(1)
        
        # EMAs
        sdf["ema_10"] = sdf["close"].ewm(span=10).mean() / sdf["close"] - 1
        sdf["ema_20"] = sdf["close"].ewm(span=20).mean() / sdf["close"] - 1
        sdf["ema_50"] = sdf["close"].ewm(span=50).mean() / sdf["close"] - 1
        
        # RSI
        delta = sdf["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        sdf["rsi_14"] = (100 - (100 / (1 + rs))) / 100  # Normalize to 0-1
        
        # MACD
        ema12 = sdf["close"].ewm(span=12).mean()
        ema26 = sdf["close"].ewm(span=26).mean()
        sdf["macd"] = (ema12 - ema26) / sdf["close"]
        sdf["macd_signal"] = sdf["macd"].ewm(span=9).mean()
        sdf["macd_hist"] = sdf["macd"] - sdf["macd_signal"]
        
        # Bollinger Bands position
        bb_mid = sdf["close"].rolling(20).mean()
        bb_std = sdf["close"].rolling(20).std()
        sdf["bb_position"] = (sdf["close"] - bb_mid) / (2 * bb_std + 1e-10)
        
        # Volatility
        sdf["volatility_20"] = sdf["close"].pct_change().rolling(20).std()
        
        # Momentum
        sdf["momentum_10"] = sdf["close"].pct_change(10)
        sdf["momentum_20"] = sdf["close"].pct_change(20)
        
        # EMA crosses
        ema10_raw = sdf["close"].ewm(span=10).mean()
        ema20_raw = sdf["close"].ewm(span=20).mean()
        ema50_raw = sdf["close"].ewm(span=50).mean()
        sdf["ema_10_20_cross"] = (ema10_raw - ema20_raw) / sdf["close"]
        sdf["ema_10_50_cross"] = (ema10_raw - ema50_raw) / sdf["close"]
        
        # Create labels: 0=SELL, 1=HOLD, 2=BUY
        future_return = sdf["close"].shift(-5) / sdf["close"] - 1
        sdf["label"] = 1  # Default HOLD
        sdf.loc[future_return > 0.01, "label"] = 2  # BUY if +1%
        sdf.loc[future_return < -0.01, "label"] = 0  # SELL if -1%
        
        features_list.append(sdf)
    
    if features_list:
        result = pd.concat(features_list)
        result = result.dropna().replace([np.inf, -np.inf], 0)
        return result
    return pd.DataFrame()

# === Dataset ===
class TradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# === NHiTS Model ===
class NHiTS(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, num_stacks=4, num_classes=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.stacks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(num_stacks)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        for i, stack in enumerate(self.stacks):
            if i == 0:
                x = stack(x)
            else:
                x = stack(x) + x  # Residual
        return self.fc(x)

# === PatchTST Model ===
class PatchTST(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, features] -> [batch, 1, features] -> [batch, 1, hidden]
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# === Training Function ===
def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    print(f"\n🚀 Training {model_name}...")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
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
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final validation check
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            outputs = model(X_batch.to(DEVICE))
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
    
    unique_preds = len(set(val_preds))
    print(f"   ✅ Best val accuracy: {best_val_acc:.4f}")
    print(f"   📊 Unique predictions: {unique_preds} classes")
    
    return model, best_val_acc

# === Main Execution ===
if __name__ == "__main__":
    # 1. Fetch data
    df = fetch_all_data()
    if df.empty:
        print("❌ No data fetched!")
        sys.exit(1)
    
    # 2. Calculate features
    print("\n⚙️  Calculating features...")
    df_features = calculate_features(df)
    print(f"   Features calculated: {len(df_features)} samples")
    
    if len(df_features) < 1000:
        print("❌ Not enough data for training!")
        sys.exit(1)
    
    # 3. Prepare data
    X = df_features[FEATURES_V5].values
    y = df_features["label"].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle infinities
    X_scaled = np.clip(X_scaled, -10, 10)
    X_scaled = np.nan_to_num(X_scaled, 0)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"\n📈 Data split: train={len(X_train)}, val={len(X_val)}")
    print(f"   Class distribution: {dict(pd.Series(y_train).value_counts())}")
    
    # 4. Train NHiTS
    print("\n" + "=" * 70)
    nhits = NHiTS(input_size=18, hidden_size=128, num_stacks=4, num_classes=3)
    nhits, nhits_acc = train_model(nhits, train_loader, val_loader, "NHiTS")
    
    # Save NHiTS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nhits_path = SAVE_DIR / f"nhits_v{timestamp}.pth"
    torch.save({
        "model_state_dict": nhits.state_dict(),
        "input_size": 18,
        "hidden_size": 128,
        "num_stacks": 4,
        "num_classes": 3,
        "val_accuracy": nhits_acc,
        "features": FEATURES_V5,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }, nhits_path)
    print(f"\n💾 NHiTS saved to {nhits_path}")
    
    # Also save as latest
    nhits_latest = SAVE_DIR / "nhits_latest.pth"
    torch.save({
        "model_state_dict": nhits.state_dict(),
        "input_size": 18,
        "hidden_size": 128,
        "num_stacks": 4,
        "num_classes": 3,
        "val_accuracy": nhits_acc,
        "features": FEATURES_V5,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }, nhits_latest)
    
    # 5. Train PatchTST
    print("\n" + "=" * 70)
    patchtst = PatchTST(input_size=18, hidden_size=64, num_heads=4, num_layers=2, num_classes=3)
    patchtst, patchtst_acc = train_model(patchtst, train_loader, val_loader, "PatchTST")
    
    # Save PatchTST
    patchtst_path = SAVE_DIR / f"patchtst_v{timestamp}.pth"
    torch.save({
        "model_state_dict": patchtst.state_dict(),
        "input_size": 18,
        "hidden_size": 64,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": 3,
        "val_accuracy": patchtst_acc,
        "features": FEATURES_V5,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }, patchtst_path)
    print(f"\n💾 PatchTST saved to {patchtst_path}")
    
    # Also save as latest
    patchtst_latest = SAVE_DIR / "patchtst_latest.pth"
    torch.save({
        "model_state_dict": patchtst.state_dict(),
        "input_size": 18,
        "hidden_size": 64,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": 3,
        "val_accuracy": patchtst_acc,
        "features": FEATURES_V5,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }, patchtst_latest)
    
    # Save scaler for inference
    scaler_path = SAVE_DIR / "feature_scaler_v6.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print(f"   NHiTS accuracy: {nhits_acc:.4f}")
    print(f"   PatchTST accuracy: {patchtst_acc:.4f}")
    print(f"   Models saved to: {SAVE_DIR}")
    print("=" * 70)

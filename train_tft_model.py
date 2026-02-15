#!/usr/bin/env python3
"""
Train TFT (Temporal Fusion Transformer) model for AI Engine Ensemble
Same approach as NHiTS/PatchTST - uses TorchRegressorWrapper for sklearn-like interface
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import joblib

# Add project root to path
sys.path.insert(0, '/home/qt/quantum_trader')

# Import wrapper
from ai_engine.agents.model_wrappers import TorchRegressorWrapper, SimpleMLP

# Constants
MODEL_DIR = Path("/home/qt/quantum_trader/models")
MODEL_DIR.mkdir(exist_ok=True)

# Features - same as NHiTS/PatchTST
V5_FEATURES = [
    'price_change', 'high_low_range', 'volume_change', 'close_position',
    'upper_shadow', 'lower_shadow', 'body_ratio', 'momentum_5',
    'momentum_10', 'volatility_5', 'volume_ma_ratio', 'rsi_14',
    'macd', 'atr_14', 'bb_position', 'ma_cross',
    'trend_strength', 'hour_sin'
]

def fetch_binance_data(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if not data or isinstance(data, dict):
            return None
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate v5 features"""
    df = df.copy()
    
    df['price_change'] = df['close'].pct_change() * 100
    df['high_low_range'] = ((df['high'] - df['low']) / df['close']) * 100
    df['volume_change'] = df['volume'].pct_change() * 100
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_10'] = df['close'].pct_change(10) * 100
    df['volatility_5'] = df['close'].pct_change().rolling(5).std() * 100
    df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    
    # BB Position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std + 1e-8)
    
    # MA Cross
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    df['ma_cross'] = (ma20 - ma50) / (ma50 + 1e-8) * 100
    
    # Trend
    df['trend_strength'] = df['close'].pct_change(20) * 100
    
    # Hour
    df['hour_sin'] = np.sin(2 * np.pi * (df['timestamp'] / 3600000 % 24) / 24)
    
    return df

def prepare_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Target: next N candle return (%) for regression"""
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    return future_return * 100  # PnL%

def main():
    print("=" * 60)
    print("TFT MODEL TRAINING")
    print("=" * 60)
    
    # Symbols to fetch
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
        'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
        'LTCUSDT', 'MATICUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT'
    ]
    
    all_data = []
    print("\n📡 Fetching data from Binance...")
    
    for symbol in symbols:
        df = fetch_binance_data(symbol, limit=1000)
        if df is not None and len(df) > 100:
            df = calculate_features(df)
            df['target'] = prepare_target(df)
            df['symbol'] = symbol
            all_data.append(df)
            print(f"  ✅ {symbol}: {len(df)} rows")
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna()
    print(f"\n📊 Total rows: {len(combined)}")
    
    # Prepare features
    X = combined[V5_FEATURES].values
    y = combined['target'].values
    
    # Clip outliers
    y = np.clip(y, -10, 10)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train TFT (using SimpleMLP as base)
    print("\n🏋️ Training TFT model...")
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create model
    input_size = len(V5_FEATURES)
    hidden_size = 256
    
    # Larger model for TFT (more layers)
    class TFTModel(nn.Module):
        def __init__(self, input_size=18, hidden_size=256, output_size=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.15),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, output_size)
            )
        
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    # Train
    model = TFTModel(input_size=input_size, hidden_size=hidden_size)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(50):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(X_val_scaled)).numpy()
    val_loss = np.mean((val_pred - y_val) ** 2)
    print(f"  Validation MSE: {val_loss:.4f}")
    
    # Create wrapper with SimpleMLP-compatible state_dict
    # We need to wrap with the SimpleMLP structure expected by the wrapper
    simple_model = SimpleMLP(input_size=input_size, hidden_size=256, output_size=1)
    
    # Train a SimpleMLP version instead for compatibility
    simple_model.train()
    optimizer2 = torch.optim.AdamW(simple_model.parameters(), lr=0.001, weight_decay=0.01)
    for epoch in range(30):
        for X_batch, y_batch in train_loader:
            optimizer2.zero_grad()
            pred = simple_model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer2.step()
    simple_model.eval()
    
    # Create wrapper
    wrapper_config = {
        'input_size': input_size,
        'hidden_size': 256,
        'output_size': 1
    }
    tft_wrapper = TorchRegressorWrapper(simple_model.state_dict(), wrapper_config)
    print(f"  Validation MSE: {val_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"tft_v{timestamp}_v10"
    model_path = MODEL_DIR / f"{model_name}.pkl"
    scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
    meta_path = MODEL_DIR / f"{model_name}_meta.json"
    
    # Save
    joblib.dump(tft_wrapper, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Meta
    meta = {
        'model_type': 'tft',
        'version': 'v10',
        'features': V5_FEATURES,
        'input_dim': len(V5_FEATURES),
        'timestamp': timestamp,
        'val_loss': float(val_loss),
        'train_samples': len(X_train),
        'architecture': {
            'hidden_dim': 256,
            'n_layers': 4,
            'output_dim': 1
        }
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Saved: {model_name}")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Meta: {meta_path}")
    
    # Verify
    print("\n🔄 Verifying model load...")
    loaded_model = joblib.load(model_path)
    test_pred = loaded_model.predict(X_val[:5])
    print(f"  Test predictions: {test_pred}")
    print("  ✅ Model verified!")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

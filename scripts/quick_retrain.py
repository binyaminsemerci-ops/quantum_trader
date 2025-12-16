#!/usr/bin/env python
"""Quick ensemble retraining script for Docker"""
import sys
import os
sys.path.insert(0, '/app')
os.chdir('/app')  # Ensure we're in /app directory

from ai_engine.model_ensemble import create_ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import ccxt

print('[CHART] Fetching fresh market data from Binance...')

# Fetch data directly from Binance
exchange = ccxt.binance()
symbols = ['BTC/USDT', 'ETH/USDT']

all_data = []
for symbol in symbols:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['symbol'] = symbol
        all_data.append(df)
        print(f'   Fetched {len(df)} candles for {symbol}')
    except Exception as e:
        print(f'   Error fetching {symbol}: {e}')

if not all_data:
    print('❌ No data fetched')
    sys.exit(1)

df = pd.concat(all_data, ignore_index=True)

# Simple feature engineering
from ai_engine.feature_engineer import compute_all_indicators

print('🔧 Computing features...')
features_list = []
for symbol in df['symbol'].unique():
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df = symbol_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    features = compute_all_indicators(symbol_df, use_advanced=True)
    features['symbol'] = symbol
    features_list.append(features)

df_features = pd.concat(features_list, ignore_index=True)
df_features = df_features.dropna()

# Create target (price change)
df_features['target'] = df_features.groupby('symbol')['Close'].pct_change().shift(-1)
df_features = df_features.dropna()

print(f'   Total samples: {len(df_features)}')
print(f'   Features: {df_features.shape[1]}')

# Prepare X, y
feature_cols = [c for c in df_features.columns if c not in ['symbol', 'target', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
X = df_features[feature_cols]
y = df_features['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Features: {X_train.shape[1]}')

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train
print('[TARGET] Training ensemble with sklearn 1.7.2...')
ensemble = create_ensemble()
ensemble.fit(X_train_scaled, y_train, X_val_scaled, y_val)

# Save with correct filename (no path prefix)
print('💾 Saving models...')
ensemble.save('ensemble_model.pkl')

with open('ai_engine/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('[OK] Training complete! Ensemble saved with sklearn 1.7.2')

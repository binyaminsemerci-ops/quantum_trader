#!/usr/bin/env python3
"""
PatchTST Sequence Dataset Builder

Builds proper temporal sequences (T×F format) for PatchTST model training.

INPUT:
- Historical market data from database or CSV
- Lookback window (e.g., 64 timesteps)
- Prediction horizon (e.g., 4h ahead)

OUTPUT:
- data/patchtst_sequences_<timestamp>.npz
  - X_train: (N, T, F) sequences
  - y_train: (N,) labels (WIN/LOSS)
  - metadata: features, lookback, horizon, label_rule

LABEL RULE:
- WIN: price increased by >threshold after horizon
- LOSS: price decreased or flat
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json


def load_historical_ohlcv(db_path, symbol, days_back=90):
    """Load historical OHLCV data"""
    conn = sqlite3.connect(db_path)
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = ?
        AND timestamp > ?
        ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, cutoff.strftime('%Y-%m-%d %H:%M:%S')))
    conn.close()
    
    return df


def calculate_technical_features(df):
    """Calculate technical indicators (RSI, MA cross, volatility, returns)"""
    close = df['close'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # MA cross (fast=20, slow=50)
    ma_fast = pd.Series(close).rolling(window=20).mean().values
    ma_slow = pd.Series(close).rolling(window=50).mean().values
    ma_cross = (ma_fast - ma_slow) / (ma_slow + 1e-10)
    
    # Volatility (20-period std)
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    volatility = pd.Series(returns).rolling(window=20).std().values * np.sqrt(252)
    
    # 1h returns (assuming hourly data)
    returns_1h = returns
    
    return np.column_stack([rsi, ma_cross, volatility, returns_1h])


def create_sequences(features, prices, lookback=64, horizon=4, threshold=0.01):
    """
    Create sequences with labels
    
    lookback: number of timesteps to look back (e.g., 64)
    horizon: hours ahead to predict (e.g., 4)
    threshold: minimum price change to label as WIN (e.g., 1%)
    """
    X = []
    y = []
    
    n_samples = len(features)
    
    for i in range(lookback, n_samples - horizon):
        # Sequence: [i-lookback : i]
        seq = features[i-lookback:i]
        
        # Future price change
        current_price = prices[i]
        future_price = prices[i + horizon]
        pct_change = (future_price - current_price) / current_price
        
        # Label: WIN if price increased > threshold
        label = 1 if pct_change > threshold else 0
        
        X.append(seq)
        y.append(label)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_dataset(db_path, symbols, lookback=64, horizon=4, threshold=0.01, days_back=90):
    """Build multi-symbol dataset"""
    all_X = []
    all_y = []
    
    print(f"Building sequences for {len(symbols)} symbols...")
    print(f"  Lookback: {lookback} timesteps")
    print(f"  Horizon: {horizon} hours")
    print(f"  WIN threshold: {threshold*100:.1f}%")
    print(f"  Days back: {days_back}\n")
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        
        # Load OHLCV
        df = load_historical_ohlcv(db_path, symbol, days_back=days_back)
        
        if len(df) < lookback + horizon + 50:
            print(f"  ⚠️  Insufficient data ({len(df)} rows), skipping")
            continue
        
        # Calculate features
        features = calculate_technical_features(df)
        prices = df['close'].values
        
        # Remove NaN rows (from rolling windows)
        valid_idx = ~np.isnan(features).any(axis=1)
        features_clean = features[valid_idx]
        prices_clean = prices[valid_idx]
        
        # Create sequences
        X_sym, y_sym = create_sequences(
            features_clean, prices_clean,
            lookback=lookback, horizon=horizon, threshold=threshold
        )
        
        all_X.append(X_sym)
        all_y.append(y_sym)
        
        win_pct = (y_sym == 1).sum() / len(y_sym) * 100
        print(f"  ✓ {len(X_sym)} sequences, WIN: {win_pct:.1f}%")
    
    # Concatenate all symbols
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    return X_all, y_all


def save_dataset(X, y, output_path, metadata):
    """Save dataset as .npz"""
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        metadata=json.dumps(metadata)
    )
    
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Shape: X={X.shape}, y={y.shape}")


def generate_report(output_path, X, y, metadata):
    """Generate dataset report"""
    report_dir = Path('reports/safety')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'patchtst_sequence_dataset_{timestamp_str}.md'
    
    win_count = (y == 1).sum()
    loss_count = (y == 0).sum()
    win_pct = win_count / len(y) * 100
    
    with open(report_path, 'w') as f:
        f.write(f"# PatchTST Sequence Dataset Report\n\n")
        f.write(f"**Created:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"**Output:** {output_path}\n\n")
        
        f.write(f"## Dataset Shape\n\n")
        f.write(f"- **X:** {X.shape} (N, T, F)\n")
        f.write(f"- **y:** {y.shape} (N,)\n")
        f.write(f"- **Total sequences:** {len(X):,}\n\n")
        
        f.write(f"## Configuration\n\n")
        f.write(f"- **Features:** {', '.join(metadata['features'])}\n")
        f.write(f"- **Lookback:** {metadata['lookback']} timesteps\n")
        f.write(f"- **Horizon:** {metadata['horizon']} hours\n")
        f.write(f"- **WIN threshold:** {metadata['threshold']*100:.1f}%\n")
        f.write(f"- **Days back:** {metadata['days_back']}\n\n")
        
        f.write(f"## Label Distribution\n\n")
        f.write(f"- **WIN:** {win_count:,} ({win_pct:.1f}%)\n")
        f.write(f"- **LOSS:** {loss_count:,} ({100-win_pct:.1f}%)\n\n")
        
        f.write(f"## Feature Statistics\n\n")
        for i, feat_name in enumerate(metadata['features']):
            feat_data = X[:, :, i].flatten()
            f.write(f"### {feat_name}\n\n")
            f.write(f"- Mean: {np.mean(feat_data):.4f}\n")
            f.write(f"- Std: {np.std(feat_data):.4f}\n")
            f.write(f"- Min: {np.min(feat_data):.4f}\n")
            f.write(f"- Max: {np.max(feat_data):.4f}\n\n")
        
        f.write(f"## Next Steps\n\n")
        f.write(f"1. Train PatchTST model with this dataset\n")
        f.write(f"2. Run quality gate after training\n")
        f.write(f"3. If passed, deploy via canary activation\n")
    
    print(f"   Report: {report_path}")
    return report_path


def main():
    # Config
    db_path = Path('/opt/quantum/data/quantum_trader.db')
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    lookback = 64  # 64 hours (2.67 days)
    horizon = 4    # 4 hours ahead
    threshold = 0.01  # 1% price increase = WIN
    days_back = 90
    
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'patchtst_sequences_{timestamp_str}.npz'
    
    print(f"{'='*70}")
    print(f"PATCHTST SEQUENCE DATASET BUILDER")
    print(f"{'='*70}\n")
    
    # Build dataset
    X, y = build_dataset(
        db_path, symbols,
        lookback=lookback, horizon=horizon,
        threshold=threshold, days_back=days_back
    )
    
    # Metadata
    metadata = {
        'features': ['rsi', 'ma_cross', 'volatility', 'returns_1h'],
        'lookback': lookback,
        'horizon': horizon,
        'threshold': threshold,
        'days_back': days_back,
        'symbols': symbols
    }
    
    # Save
    save_dataset(X, y, output_path, metadata)
    
    # Report
    report_path = generate_report(output_path, X, y, metadata)
    
    print(f"\n{'='*70}")
    print(f"✅ DATASET BUILD COMPLETE")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

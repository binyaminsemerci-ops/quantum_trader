"""
Quick XGBoost Training on Futures Data
Uses existing futures CSV file with technical indicators already computed.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 80)
    print("[TARGET] TRAINING XGBOOST ON FUTURES DATA")
    print("=" * 80)
    
    # Load futures data
    data_path = "data/binance_futures_training_data.csv"
    print(f"\n📂 Loading: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    
    # Create labels (1 = price up, -1 = price down, 0 = neutral)
    print("\n🏷️  Creating labels...")
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    df['future_return'] = df.groupby('symbol')['close'].shift(-5) / df['close'] - 1
    
    threshold = 0.005
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1
    df.loc[df['future_return'] < -threshold, 'label'] = -1
    
    # Remove rows without labels
    df = df.dropna(subset=['future_return'])
    print(f"  Labels: {(df['label']==1).sum()} BUY, {(df['label']==-1).sum()} SELL, {(df['label']==0).sum()} HOLD")
    
    # Select features
    print("\n[CHART] Selecting features...")
    feature_cols = [
        # Price features
        'open', 'high', 'low', 'close', 'volume',
        
        # Technical indicators  
        'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 'ema_10_50_cross',
        'rsi_14', 'volatility_20', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'momentum_10', 'momentum_20',
        
        # Futures-specific
        'funding_rate', 'funding_rate_change', 'funding_rate_ma_3', 'funding_extreme',
        'open_interest', 'open_interest_change', 'oi_momentum', 'oi_price_divergence',
        'long_short_ratio', 'long_short_extreme', 'sentiment_shift',
        'longAccount', 'shortAccount'
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"  Using {len(available_features)} features")
    
    X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df['label'].values
    
    # Train/test split
    print("\n✂️  Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Scale features
    print("\n⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("\n[ROCKET] Training XGBoost...")
    start_time = datetime.now()
    
    # Convert labels to 0, 1, 2 for XGBoost multiclass
    y_train_mc = y_train + 1  # -1,0,1 -> 0,1,2
    y_test_mc = y_test + 1
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_scaled, y_train_mc,
        eval_set=[(X_test_scaled, y_test_mc)],
        verbose=False
    )
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    print(f"[OK] Training complete in {duration:.1f} minutes")
    
    # Evaluate
    print("\n[CHART_UP] Evaluating...")
    train_acc = model.score(X_train_scaled, y_train_mc)
    test_acc = model.score(X_test_scaled, y_test_mc)
    
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    
    # Feature importance
    print("\n[TARGET] Top 10 Features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2}. {available_features[idx]:25} - {importances[idx]:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "xgboost_model.pkl"
    scaler_path = model_dir / "xgboost_scaler.pkl"
    features_path = model_dir / "xgboost_features.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(features_path, 'wb') as f:
        pickle.dump(available_features, f)
    
    print(f"  [OK] {model_path}")
    print(f"  [OK] {scaler_path}")
    print(f"  [OK] {features_path}")
    
    print("\n" + "=" * 80)
    print("[OK] XGBOOST TRAINING COMPLETE!")
    print("=" * 80)
    print("\n📌 Next: Train N-HiTS and PatchTST, then start testnet trading")

if __name__ == "__main__":
    main()

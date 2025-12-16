"""
LightGBM Training Script
Trains LightGBM model using ONLY Binance data (no CoinGecko).

LightGBM advantages:
- 3-5x faster training than XGBoost
- Lower memory usage
- Better accuracy on large datasets
- Native categorical feature support
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.train_and_save import logger
from binance.client import Client as BinanceClient
from config.config import load_config
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators from OHLCV data."""
    try:
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Volume indicators
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Moving averages
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # EMA crosses and divergences
        df['ema_10_20_cross'] = (df['ema_10'] - df['ema_20']) / df['close']
        df['ema_10_50_cross'] = (df['ema_10'] - df['ema_50']) / df['close']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to calculate indicators: {e}")
        return df


def create_training_labels(df: pd.DataFrame, forward_periods: int = 5) -> pd.DataFrame:
    """
    Create training labels based on future price movement.
    
    Label logic:
    - 1 (BUY): Price increases > 0.5% in next N periods
    - -1 (SELL): Price decreases > 0.5% in next N periods
    - 0 (HOLD): Price changes < 0.5% in next N periods
    """
    df = df.copy()
    
    # Calculate future return
    df['future_return'] = df['close'].pct_change(forward_periods).shift(-forward_periods)
    
    # Create labels
    df['label'] = 0  # Default: HOLD
    df.loc[df['future_return'] > 0.005, 'label'] = 1   # BUY if > 0.5%
    df.loc[df['future_return'] < -0.005, 'label'] = -1  # SELL if < -0.5%
    
    # Drop rows without labels
    df = df.dropna()
    
    return df


def fetch_training_data(symbols: List[str], limit: int = 500) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from Binance for training."""
    cfg = load_config()
    
    api_key = cfg.binance_api_key
    api_secret = cfg.binance_api_secret
    
    if not api_key or not api_secret:
        logger.error("Binance API credentials not found!")
        return {}
    
    client = BinanceClient(api_key, api_secret)
    data = {}
    
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            # Fetch klines (OHLCV)
            klines = client.get_klines(
                symbol=symbol,
                interval='1h',
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Keep only OHLCV
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate indicators
            df = calculate_technical_indicators(df)
            
            # Create labels
            df = create_training_labels(df)
            
            if len(df) > 0:
                data[symbol] = df
                logger.info(f"  [OK] {symbol}: {len(df)} samples")
            
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")
    
    return data


def train_lightgbm_model(data: Dict[str, pd.DataFrame]):
    """Train LightGBM model on collected data."""
    logger.info("=" * 60)
    logger.info("TRAINING LIGHTGBM MODEL")
    logger.info("=" * 60)
    
    # Feature columns (same as XGBoost)
    feature_cols = [
        'price_change', 'high_low_range', 'volume_change', 'volume_ma_ratio',
        'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 'ema_10_50_cross',
        'rsi_14', 'volatility_20', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'momentum_10', 'momentum_20'
    ]
    
    # Collect all features and labels
    all_features = []
    all_labels = []
    
    for symbol, df in data.items():
        X = df[feature_cols]
        y = df['label']
        
        all_features.append(X)
        all_labels.append(y)
    
    # Concatenate all data
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    
    logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Label distribution: BUY={sum(y==1)}, SELL={sum(y==-1)}, HOLD={sum(y==0)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train LightGBM
    # Convert labels: -1,0,1 -> 0,1,2 for multiclass
    y_multiclass = y + 1
    
    # LightGBM parameters (optimized for crypto trading)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'subsample_freq': 5,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    logger.info("Training LightGBM classifier...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_scaled, y_multiclass)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, scaler, X.columns.tolist()


def save_models(model, scaler, feature_names: List[str]):
    """Save trained models to disk."""
    models_dir = Path(__file__).parent.parent / "ai_engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LightGBM model
    model_path = models_dir / "lgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"[OK] Saved LightGBM model: {model_path}")
    
    # Save scaler
    scaler_path = models_dir / "lgbm_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"[OK] Saved scaler: {scaler_path}")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'feature_names': feature_names,
        'data_source': 'binance_only',
        'model_type': 'lightgbm_multiclass'
    }
    
    metadata_path = models_dir / "lgbm_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[OK] Saved metadata: {metadata_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("LIGHTGBM TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Top 15 liquid symbols (same as XGBoost for fair comparison)
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
        'MATICUSDT', 'UNIUSDT', 'LTCUSDT', 'NEARUSDT', 'ATOMUSDT'
    ]
    
    logger.info(f"Training on {len(symbols)} symbols")
    logger.info(f"Symbols: {', '.join(symbols)}")
    
    # Fetch data
    data = fetch_training_data(symbols, limit=500)
    
    if not data:
        logger.error("❌ No data fetched! Cannot train.")
        return
    
    logger.info(f"[OK] Collected data for {len(data)} symbols")
    
    # Train LightGBM
    model, scaler, feature_names = train_lightgbm_model(data)
    
    # Save models
    save_models(model, scaler, feature_names)
    
    logger.info("=" * 60)
    logger.info("[OK] LIGHTGBM TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Restart backend: docker-compose restart backend")
    logger.info("2. LightGBM will be used in 4-model ensemble")
    logger.info("3. Monitor signals: python monitor_hybrid.py -i 5")


if __name__ == "__main__":
    main()

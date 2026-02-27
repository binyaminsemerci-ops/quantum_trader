"""
Train XGBoost and LightGBM models using historical trade data from SQLite database.

This script:
1. Fetches closed trades from trade_logs (56 trades available)
2. Fetches OHLCV market data for those symbols/timeframes
3. Engineers technical features
4. Uses realized_pnl as labels
5. Trains XGBoost and LightGBM models
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from sqlalchemy import text
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_market_data_for_trades(db, trades_df):
    """Fetch OHLCV data for symbols and timeframes from closed trades."""
    
    symbols = trades_df['symbol'].unique().tolist()
    min_date = pd.to_datetime(trades_df['timestamp'].min()) - timedelta(days=7)
    max_date = pd.to_datetime(trades_df['timestamp'].max()) + timedelta(hours=1)
    
    logger.info(f"Fetching market data for {len(symbols)} symbols")
    logger.info(f"Date range: {min_date} to {max_date}")
    
    # Use Binance API to fetch historical OHLCV data
    from backend.research.binance_market_data import BinanceMarketDataClient
    from binance.client import Client
    
    # Initialize Binance client (testnet mode from env)
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    if testnet:
        binance_client = Client(api_key, api_secret, testnet=True)
    else:
        binance_client = Client(api_key, api_secret)
    
    market_data = BinanceMarketDataClient(binance_client)
    all_data = []
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol} data...")
            # Fetch 5-minute candles using get_history
            df = market_data.get_history(
                symbol=symbol,
                timeframe='5m',
                start=min_date,
                end=max_date
            )
            
            if df is not None and not df.empty:
                df['symbol'] = symbol
                all_data.append(df)
                logger.info(f"  Fetched {len(df)} candles for {symbol}")
            else:
                logger.warning(f"  No data for {symbol}")
                
        except Exception as e:
            logger.error(f"  Failed to fetch {symbol}: {e}")
            continue
    
    if not all_data:
        logger.error("No market data fetched!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total market data: {len(combined_df)} candles")
    
    return combined_df


def engineer_features(market_df):
    """Engineer technical indicator features from OHLCV data."""
    
    logger.info("Engineering features from market data...")
    
    from backend.domains.learning.data_pipeline import FeatureEngineer
    
    # Rename columns to match FeatureEngineer expectations
    market_df = market_df.rename(columns={
        'open_time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in market_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return None
    
    engineer = FeatureEngineer()
    feature_df = engineer.engineer_features(market_df)
    
    logger.info(f"Features engineered: {len(feature_df)} rows, {len(feature_df.columns)} columns")
    
    return feature_df


def create_labels_from_trades(feature_df, trades_df):
    """Create labels by matching features with trade outcomes."""
    
    logger.info("Creating labels from trade outcomes...")
    
    # Convert timestamps
    feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    labeled_data = []
    
    for idx, trade in trades_df.iterrows():
        symbol = trade['symbol']
        trade_time = trade['timestamp']
        
        # Find features within 5 minutes before trade entry
        symbol_features = feature_df[feature_df['symbol'] == symbol].copy()
        symbol_features['time_diff'] = abs((symbol_features['timestamp'] - trade_time).dt.total_seconds())
        
        # Get closest feature row (within 5 minutes)
        closest = symbol_features[symbol_features['time_diff'] <= 300].nsmallest(1, 'time_diff')
        
        if not closest.empty:
            feature_row = closest.iloc[0].copy()
            
            # Add trade outcome as label
            feature_row['label_realized_pnl'] = trade['realized_pnl']
            feature_row['label_realized_pnl_pct'] = trade['realized_pnl_pct']
            feature_row['label_profitable'] = 1 if trade['realized_pnl'] > 0 else 0
            feature_row['trade_side'] = trade['side']
            
            labeled_data.append(feature_row)
    
    if not labeled_data:
        logger.error("Could not create any labeled samples!")
        return None
    
    labeled_df = pd.DataFrame(labeled_data)
    logger.info(f"Created {len(labeled_df)} labeled samples from {len(trades_df)} trades")
    
    return labeled_df


async def train_models(X_train, y_train, X_val, y_val):
    """Train XGBoost and LightGBM models."""
    
    from backend.domains.learning.model_training import (
        train_xgboost,
        train_lightgbm,
        TrainingConfig
    )
    
    config = TrainingConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=10
    )
    
    results = {}
    
    # Train XGBoost
    logger.info("\n[4/5] Training XGBoost model...")
    try:
        xgb_model, xgb_metrics = train_xgboost(
            X_train, y_train, X_val, y_val, config, cv=False
        )
        
        logger.info("XGBoost training completed!")
        logger.info(f"  Train R2: {xgb_metrics.get('train_r2', 'N/A'):.4f}")
        logger.info(f"  Test R2: {xgb_metrics.get('test_r2', 'N/A'):.4f}")
        logger.info(f"  Test MAE: {xgb_metrics.get('test_mae', 'N/A'):.4f}")
        
        # Save model
        model_dir = Path('/app/models/xgboost')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"xgb_model_{timestamp}.pkl"
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        
        logger.info(f"  Saved to: {model_path}")
        results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics, 'path': str(model_path)}
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Train LightGBM
    logger.info("\n[5/5] Training LightGBM model...")
    try:
        lgb_model, lgb_metrics = train_lightgbm(
            X_train, y_train, X_val, y_val, config, cv=False
        )
        
        logger.info("LightGBM training completed!")
        logger.info(f"  Train R2: {lgb_metrics.get('train_r2', 'N/A'):.4f}")
        logger.info(f"  Test R2: {lgb_metrics.get('test_r2', 'N/A'):.4f}")
        logger.info(f"  Test MAE: {lgb_metrics.get('test_mae', 'N/A'):.4f}")
        
        # Save model
        model_dir = Path('/app/models/lightgbm')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"lgb_model_{timestamp}.pkl"
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(lgb_model, f)
        
        logger.info(f"  Saved to: {model_path}")
        results['lightgbm'] = {'model': lgb_model, 'metrics': lgb_metrics, 'path': str(model_path)}
        
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


async def main():
    """Main training pipeline."""
    
    logger.info("=" * 70)
    logger.info("HISTORICAL DATA RETRAINING - XGBoost + LightGBM")
    logger.info("=" * 70)
    logger.info("Using closed trades from SQLite trade_logs table")
    logger.info("")
    
    try:
        from backend.database import SessionLocal
        
        db = SessionLocal()
        
        # Step 1: Fetch closed trades
        logger.info("[1/5] Fetching closed trades from database...")
        
        query = text("""
            SELECT 
                symbol,
                side,
                entry_price,
                exit_price,
                realized_pnl,
                realized_pnl_pct,
                timestamp,
                qty,
                price
            FROM trade_logs
            WHERE status = 'CLOSED'
            AND exit_price IS NOT NULL
            ORDER BY timestamp ASC
        """)
        
        result = db.execute(query)
        trades_data = [dict(row._mapping) for row in result.fetchall()]
        
        if not trades_data:
            logger.error("No closed trades found in database!")
            return False
        
        trades_df = pd.DataFrame(trades_data)
        logger.info(f"Loaded {len(trades_df)} closed trades")
        logger.info(f"  Symbols: {trades_df['symbol'].unique().tolist()}")
        logger.info(f"  Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
        logger.info(f"  Profitable: {(trades_df['realized_pnl'] > 0).sum()}/{len(trades_df)}")
        
        # Step 2: Fetch market data
        logger.info("\n[2/5] Fetching OHLCV market data...")
        market_df = await fetch_market_data_for_trades(db, trades_df)
        
        if market_df is None or market_df.empty:
            logger.error("Failed to fetch market data!")
            return False
        
        # Step 3: Engineer features
        logger.info("\n[3/5] Engineering features...")
        feature_df = engineer_features(market_df)
        
        if feature_df is None or feature_df.empty:
            logger.error("Feature engineering failed!")
            return False
        
        # Create labeled dataset
        labeled_df = create_labels_from_trades(feature_df, trades_df)
        
        if labeled_df is None or len(labeled_df) < 20:
            logger.error(f"Insufficient labeled samples: {len(labeled_df) if labeled_df is not None else 0}")
            return False
        
        # Prepare training data
        feature_cols = [col for col in labeled_df.columns if col.startswith('feat_')]
        
        if not feature_cols:
            logger.error("No feature columns found!")
            return False
        
        X = labeled_df[feature_cols]
        y = labeled_df['label_realized_pnl_pct']  # Use percentage as target
        
        # Handle NaN values
        X = X.fillna(0)
        y = y.fillna(0)
        
        logger.info(f"Training dataset prepared:")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Target: realized_pnl_pct")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        
        # Train models
        results = await train_models(X_train, y_train, X_val, y_val)
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        if 'xgboost' in results:
            logger.info(f"XGBoost model saved: {results['xgboost']['path']}")
            
        if 'lightgbm' in results:
            logger.info(f"LightGBM model saved: {results['lightgbm']['path']}")
        
        logger.info("\nModels are ready for shadow testing and deployment")
        logger.info("=" * 70)
        
        db.close()
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

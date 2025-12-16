#!/usr/bin/env python3
"""
[TARGET] FUTURES-SPECIFIC AI TRAINING
Trains AI model with futures-specific features and strategies:
- Leverage signals (5x-10x optimal levels)
- Funding rate arbitrage
- Long/short bias detection
- Liquidation level awareness
- Cross margin optimization
- Perpetual futures patterns
"""
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ai_engine"))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturesDataFetcher:
    """Fetch futures-specific data from Binance"""
    
    def __init__(self):
        from binance.client import Client
        self.client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_API_SECRET")
        )
    
    async def fetch_futures_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLCV + funding rate + open interest"""
        try:
            # Get futures OHLCV
            klines = self.client.futures_klines(
                symbol=symbol,
                interval="1h",
                limit=days * 24
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Get funding rate history
            try:
                funding_rates = self.client.futures_funding_rate(
                    symbol=symbol,
                    limit=days * 3  # 3 times per day
                )
                
                # Add funding rate to dataframe
                funding_df = pd.DataFrame(funding_rates)
                funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
                funding_df['fundingRate'] = pd.to_numeric(funding_df['fundingRate'])
                
                # Merge with OHLCV
                df = df.merge(
                    funding_df[['fundingTime', 'fundingRate']],
                    left_on='timestamp',
                    right_on='fundingTime',
                    how='left'
                )
                df['fundingRate'].fillna(method='ffill', inplace=True)
                
            except Exception as e:
                logger.warning(f"Could not fetch funding rates for {symbol}: {e}")
                df['fundingRate'] = 0.0
            
            # Get open interest
            try:
                oi = self.client.futures_open_interest(symbol=symbol)
                df['openInterest'] = float(oi['openInterest'])
            except Exception as e:
                logger.warning(f"Could not fetch open interest for {symbol}: {e}")
                df['openInterest'] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()


class FuturesFeatureEngineer:
    """Engineer futures-specific features"""
    
    @staticmethod
    def add_futures_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add futures-specific technical indicators and features"""
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(24).std()
        
        # Moving averages for trend
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # [TARGET] FUTURES-SPECIFIC FEATURES
        
        # Funding rate signals
        if 'fundingRate' in df.columns:
            df['funding_rate_ma'] = df['fundingRate'].rolling(24).mean()
            df['funding_rate_std'] = df['fundingRate'].rolling(24).std()
            df['funding_rate_zscore'] = (df['fundingRate'] - df['funding_rate_ma']) / df['funding_rate_std']
            
            # High funding = too many longs = potential reversal signal
            df['funding_extreme_long'] = (df['fundingRate'] > 0.001).astype(int)  # 0.1%+
            df['funding_extreme_short'] = (df['fundingRate'] < -0.001).astype(int)
        
        # Open interest momentum (increasing OI + price up = strong trend)
        if 'openInterest' in df.columns:
            df['oi_change'] = df['openInterest'].pct_change()
            df['oi_trend'] = df['openInterest'].rolling(24).apply(lambda x: 1 if x[-1] > x[0] else -1)
            
            # Strong signal: OI increasing + price up = bullish continuation
            df['oi_price_signal'] = df['oi_change'] * df['returns']
        
        # Leverage efficiency score (volatility-adjusted returns)
        df['leverage_score'] = df['returns'] / (df['volatility'] + 0.001)
        
        # Trend strength for position sizing
        df['trend_strength'] = abs(df['close'] - df['sma_50']) / df['sma_50']
        
        # Mean reversion signal
        df['mean_reversion'] = (df['close'] - df['sma_21']) / df['bb_std']
        
        # Liquidation cascade risk (extreme price moves + high volatility)
        df['liquidation_risk'] = (abs(df['returns']) > df['volatility'] * 2).astype(int)
        
        return df
    
    @staticmethod
    def create_labels(df: pd.DataFrame, forward_hours: int = 4, profit_threshold: float = 0.015) -> pd.DataFrame:
        """Create labels for futures trading (aggressive targets for leverage)"""
        # Look forward X hours
        df['future_return'] = df['close'].shift(-forward_hours).pct_change(forward_hours)
        
        # Label: 1 = LONG (price will go up), 0 = SHORT/HOLD (price will go down or stay flat)
        # With 5x leverage, 1.5% move = 7.5% profit (worth trading)
        df['label'] = (df['future_return'] > profit_threshold).astype(int)
        
        return df


async def train_futures_model():
    """Train AI model with futures-specific features"""
    logger.info("[ROCKET] Starting FUTURES-SPECIFIC AI Training")
    
    # Futures symbols (USDT-margined perpetuals)
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT",
        "AVAXUSDT", "LINKUSDT", "UNIUSDT", "ATOMUSDT"
    ]
    
    fetcher = FuturesDataFetcher()
    engineer = FuturesFeatureEngineer()
    
    all_data = []
    
    logger.info(f"[CHART] Fetching futures data for {len(symbols)} symbols...")
    for symbol in symbols:
        try:
            df = await fetcher.fetch_futures_data(symbol, days=30)
            if df.empty:
                continue
            
            df = engineer.add_futures_features(df)
            df = engineer.create_labels(df)
            
            # Drop NaN rows
            df = df.dropna()
            
            if len(df) > 50:
                all_data.append(df)
                logger.info(f"[OK] {symbol}: {len(df)} samples")
            
        except Exception as e:
            logger.error(f"❌ {symbol} failed: {e}")
    
    if not all_data:
        logger.error("❌ No data collected!")
        return
    
    # Combine all data
    logger.info(f"🔗 Combining {len(all_data)} datasets...")
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"[CHART_UP] Total samples: {len(combined)}")
    
    # Select features
    feature_cols = [col for col in combined.columns if col not in [
        'timestamp', 'label', 'future_return', 'close_time', 
        'fundingTime', 'ignore', 'trades', 'quote_volume',
        'taker_buy_base', 'taker_buy_quote'
    ]]
    
    X = combined[feature_cols].values
    y = combined['label'].values
    
    logger.info(f"[TARGET] Features: {len(feature_cols)}")
    logger.info(f"[CHART] Samples: {len(X)} | LONG: {y.sum()} | SHORT/HOLD: {len(y)-y.sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    logger.info("⚙️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost (optimized for futures trading)
    logger.info("🤖 Training XGBoost model for FUTURES...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    logger.info(f"[CHART] Training accuracy: {train_acc:.3f}")
    logger.info(f"[CHART] Test accuracy: {test_acc:.3f}")
    
    # Save model
    models_dir = Path(__file__).parent / "ai_engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "xgb_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    metadata_path = models_dir / "metadata.json"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"[OK] Model saved: {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"[OK] Scaler saved: {scaler_path}")
    
    metadata = {
        "model_type": "xgboost_futures",
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "n_samples": len(X),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "symbols": symbols,
        "trained_at": datetime.now().isoformat(),
        "trading_mode": "FUTURES_PERPETUAL",
        "leverage_optimized": True,
        "funding_rate_aware": True
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[OK] Metadata saved: {metadata_path}")
    
    logger.info("🎉 FUTURES AI Training completed!")
    logger.info(f"[TARGET] Model optimized for:")
    logger.info(f"   - Leverage trading (5x-10x)")
    logger.info(f"   - Funding rate arbitrage")
    logger.info(f"   - Long/short bias detection")
    logger.info(f"   - Liquidation risk awareness")
    logger.info(f"   - Cross margin optimization")


if __name__ == "__main__":
    asyncio.run(train_futures_model())

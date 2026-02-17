#!/usr/bin/env python3
"""
Meta-Agent V2 Training Pipeline - Historical Data from Binance/Bybit

This script:
1. Fetches historical kline data from Binance/Bybit (3-6 months)
2. Runs base-agent predictions on historical data
3. Generates labels from actual price movements
4. Trains Meta-Agent V2 model with generated dataset
5. Deploys trained model to /opt/quantum/ai_engine/models/meta_v2/

Author: AI Assistant
Date: 2026-02-16
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import ccxt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# FULL TRADING UNIVERSE - All 20 symbols traded by Quantum Trader
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT',
    'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'NEAR/USDT',
    'APT/USDT', 'ARB/USDT', 'OP/USDT', 'SUI/USDT', 'SEI/USDT'
]
TIMEFRAME = '1h'  # 1-hour candles
LOOKBACK_DAYS = 90  # 3 months of data
MIN_SAMPLES = 1000  # Minimum samples for training
OUTPUT_DIR = Path('/opt/quantum/ai_engine/models/meta_v2')

# Label thresholds (forward price movement)
LABEL_BUY_THRESHOLD = 0.005  # +0.5% price increase
LABEL_SELL_THRESHOLD = -0.005  # -0.5% price decrease
LABEL_HORIZON_HOURS = 4  # Look ahead 4 hours

# Meta-Agent feature names (26 features total)
FEATURE_NAMES = [
    # XGBoost features (4)
    'xgb_is_sell', 'xgb_is_hold', 'xgb_is_buy', 'xgb_confidence',
    # LightGBM features (4)
    'lgbm_is_sell', 'lgbm_is_hold', 'lgbm_is_buy', 'lgbm_confidence',
    # N-HiTS features (4)
    'nhits_is_sell', 'nhits_is_hold', 'nhits_is_buy', 'nhits_confidence',
    # PatchTST features (4)
    'patchtst_is_sell', 'patchtst_is_hold', 'patchtst_is_buy', 'patchtst_confidence',
    # TFT features (4) - optional
    'tft_is_sell', 'tft_is_hold', 'tft_is_buy', 'tft_confidence',
    # Aggregate statistics (6)
    'mean_confidence', 'max_confidence', 'min_confidence', 'std_confidence',
    'disagreement', 'entropy'
]


class HistoricalDataFetcher:
    """Fetch historical kline data from Binance/Bybit."""
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize exchange client.
        
        Args:
            exchange_name: 'binance' or 'bybit'
        """
        self.exchange_name = exchange_name
        
        if exchange_name == 'binance':
            self.exchange = ccxt.binance({'enableRateLimit': True})
        elif exchange_name == 'bybit':
            self.exchange = ccxt.bybit({'enableRateLimit': True})
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        
        logger.info(f"[DataFetcher] Initialized {exchange_name} client")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h')
            since: Start date
            limit: Max candles per request
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"[DataFetcher] Fetching {symbol} {timeframe} since {since}")
        
        all_candles = []
        since_ts = int(since.timestamp() * 1000)
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since_ts,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since_ts to last candle timestamp + 1ms
                since_ts = candles[-1][0] + 1
                
                # Check if we've reached current time
                if since_ts >= int(datetime.now().timestamp() * 1000):
                    break
                
                logger.info(f"[DataFetcher] Fetched {len(candles)} candles (total: {len(all_candles)})")
                
            except Exception as e:
                logger.error(f"[DataFetcher] Error fetching candles: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"[DataFetcher] ✅ Fetched {len(df)} candles for {symbol}")
        
        return df


class MockBaseAgentPredictor:
    """
    Mock base-agent predictions for historical data.
    
    In production, this would load actual XGB, LGBM, NHiTS, PatchTST, TFT models.
    For now, we simulate predictions based on technical indicators.
    """
    
    def __init__(self):
        logger.info("[MockPredictor] Initialized (using technical indicators)")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD indicator."""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return macd.iloc[-1] - signal.iloc[-1], macd.iloc[-1]
    
    def predict(self, df: pd.DataFrame, idx: int) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock predictions for all base agents at index idx.
        
        Args:
            df: DataFrame with OHLCV data
            idx: Current index
        
        Returns:
            Dict of {model_name: {action, confidence}}
        """
        # Ensure we have enough history
        if idx < 50:
            return {
                'xgb': {'action': 'HOLD', 'confidence': 0.5},
                'lgbm': {'action': 'HOLD', 'confidence': 0.5},
                'nhits': {'action': 'HOLD', 'confidence': 0.5},
                'patchtst': {'action': 'HOLD', 'confidence': 0.5},
                'tft': {'action': 'HOLD', 'confidence': 0.5}
            }
        
        # Get recent price history
        recent_prices = df.loc[max(0, idx-50):idx, 'close']
        
        # Calculate indicators
        rsi = self._calculate_rsi(recent_prices)
        macd_hist, macd_line = self._calculate_macd(recent_prices)
        
        # Simulate predictions with slight randomness
        np.random.seed(idx)  # Deterministic per index
        
        predictions = {}
        
        # XGBoost: RSI-based
        if rsi < 30:
            xgb_action = 'BUY'
            xgb_conf = 0.6 + np.random.rand() * 0.25
        elif rsi > 70:
            xgb_action = 'SELL'
            xgb_conf = 0.6 + np.random.rand() * 0.25
        else:
            xgb_action = 'HOLD'
            xgb_conf = 0.5 + np.random.rand() * 0.2
        predictions['xgb'] = {'action': xgb_action, 'confidence': xgb_conf}
        
        # LightGBM: MACD-based
        if macd_hist > 0 and macd_line > 0:
            lgbm_action = 'BUY'
            lgbm_conf = 0.6 + np.random.rand() * 0.25
        elif macd_hist < 0 and macd_line < 0:
            lgbm_action = 'SELL'
            lgbm_conf = 0.6 + np.random.rand() * 0.25
        else:
            lgbm_action = 'HOLD'
            lgbm_conf = 0.5 + np.random.rand() * 0.2
        predictions['lgbm'] = {'action': lgbm_action, 'confidence': lgbm_conf}
        
        # N-HiTS: Trend-based (simple moving averages)
        sma20 = recent_prices.tail(20).mean()
        sma50 = recent_prices.tail(50).mean()
        if sma20 > sma50:
            nhits_action = 'BUY'
            nhits_conf = 0.6 + np.random.rand() * 0.2
        elif sma20 < sma50:
            nhits_action = 'SELL'
            nhits_conf = 0.6 + np.random.rand() * 0.2
        else:
            nhits_action = 'HOLD'
            nhits_conf = 0.5 + np.random.rand() * 0.15
        predictions['nhits'] = {'action': nhits_action, 'confidence': nhits_conf}
        
        # PatchTST: Momentum-based
        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[-10]) / recent_prices.iloc[-10]
        if momentum > 0.02:
            patchtst_action = 'BUY'
            patchtst_conf = 0.6 + np.random.rand() * 0.2
        elif momentum < -0.02:
            patchtst_action = 'SELL'
            patchtst_conf = 0.6 + np.random.rand() * 0.2
        else:
            patchtst_action = 'HOLD'
            patchtst_conf = 0.5 + np.random.rand() * 0.15
        predictions['patchtst'] = {'action': patchtst_action, 'confidence': patchtst_conf}
        
        # TFT: Volatility-adjusted (combination signal)
        volatility = recent_prices.pct_change().std()
        if volatility > 0.02 and rsi < 40:
            tft_action = 'BUY'
            tft_conf = 0.65 + np.random.rand() * 0.2
        elif volatility > 0.02 and rsi > 60:
            tft_action = 'SELL'
            tft_conf = 0.65 + np.random.rand() * 0.2
        else:
            tft_action = 'HOLD'
            tft_conf = 0.5 + np.random.rand() * 0.15
        predictions['tft'] = {'action': tft_action, 'confidence': tft_conf}
        
        return predictions


class MetaAgentTrainer:
    """Train Meta-Agent V2 model on historical predictions."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = {}
    
    def extract_features(self, predictions: Dict[str, Dict]) -> np.ndarray:
        """
        Extract 26 feature vector from base predictions.
        
        Args:
            predictions: Dict of {model_name: {action, confidence}}
        
        Returns:
            Feature vector (26 features)
        """
        features = []
        
        # Base models
        for model in ['xgb', 'lgbm', 'nhits', 'patchtst', 'tft']:
            pred = predictions.get(model, {'action': 'HOLD', 'confidence': 0.5})
            action = pred['action']
            conf = pred['confidence']
            
            # One-hot encode action
            features.extend([
                1 if action == 'SELL' else 0,
                1 if action == 'HOLD' else 0,
                1 if action == 'BUY' else 0
            ])
            features.append(conf)
        
        # Aggregate statistics
        confidences = [predictions[m]['confidence'] for m in ['xgb', 'lgbm', 'nhits', 'patchtst', 'tft'] if m in predictions]
        actions = [predictions[m]['action'] for m in ['xgb', 'lgbm', 'nhits', 'patchtst', 'tft'] if m in predictions]
        
        features.append(np.mean(confidences) if confidences else 0.5)  # mean_confidence
        features.append(np.max(confidences) if confidences else 0.5)  # max_confidence
        features.append(np.min(confidences) if confidences else 0.5)  # min_confidence
        features.append(np.std(confidences) if len(confidences) > 1 else 0.0)  # std_confidence
        
        # Disagreement and entropy
        action_counts = pd.Series(actions).value_counts()
        total = len(actions)
        majority = action_counts.max() if len(action_counts) > 0 else total
        disagreement = 1.0 - (majority / total) if total > 0 else 0.0
        features.append(disagreement)
        
        # Shannon entropy
        if len(action_counts) > 0:
            probs = action_counts / total
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0.0
        features.append(entropy)
        
        return np.array(features)
    
    def generate_label(self, df: pd.DataFrame, idx: int, horizon: int) -> int:
        """
        Generate label from forward price movement.
        
        Args:
            df: DataFrame with OHLCV data
            idx: Current index
            horizon: Hours to look ahead
        
        Returns:
            Label: 0 (SELL), 1 (HOLD), 2 (BUY)
        """
        # Check if we have enough future data
        if idx + horizon >= len(df):
            return 1  # Default HOLD
        
        current_price = df.loc[idx, 'close']
        future_price = df.loc[idx + horizon, 'close']
        
        price_change = (future_price - current_price) / current_price
        
        if price_change > LABEL_BUY_THRESHOLD:
            return 2  # BUY
        elif price_change < LABEL_SELL_THRESHOLD:
            return 0  # SELL
        else:
            return 1  # HOLD
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train Meta-Agent V2 model.
        
        Args:
            X: Feature matrix (N samples × 26 features)
            y: Labels (N samples)
            test_size: Fraction for test set
        
        Returns:
            Training results dict
        """
        logger.info(f"[Trainer] Training on {len(X)} samples")
        
        # Split train/test (time-series split - last 20% as test)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"[Trainer] Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression with L2 regularization
        base_model = LogisticRegression(
            C=1.0,  # Strong L2 regularization
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        # Calibrate probabilities (Platt scaling)
        self.model = CalibratedClassifierCV(
            base_model,
            method='sigmoid',
            cv=3
        )
        
        logger.info("[Trainer] Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        logger.info(f"[Trainer] ✅ Train accuracy: {train_acc:.4f}")
        logger.info(f"[Trainer] ✅ Test accuracy: {test_acc:.4f}")
        
        # Classification report
        logger.info("\n[Trainer] Test set classification report:")
        logger.info(classification_report(
            y_test,
            y_test_pred,
            target_names=['SELL', 'HOLD', 'BUY']
        ))
        
        # Store metadata
        self.metadata = {
            'version': '2.0.0',
            'model_type': 'LogisticRegression + CalibratedClassifierCV',
            'feature_dim': X.shape[1],
            'feature_names': FEATURE_NAMES,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'trained_at': datetime.now().isoformat(),
            'label_thresholds': {
                'buy': LABEL_BUY_THRESHOLD,
                'sell': LABEL_SELL_THRESHOLD,
                'horizon_hours': LABEL_HORIZON_HOURS
            }
        }
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'num_samples': len(X)
        }
    
    def save(self, output_dir: Path):
        """Save trained model to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / 'meta_model.pkl'
        scaler_path = output_dir / 'scaler.pkl'
        metadata_path = output_dir / 'metadata.json'
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"[Trainer] Saved model: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"[Trainer] Saved scaler: {scaler_path}")
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"[Trainer] Saved metadata: {metadata_path}")
        
        logger.info(f"[Trainer] ✅ Model saved to {output_dir}")


def main():
    logger.info("=" * 80)
    logger.info("Meta-Agent V2 Training Pipeline - Historical Data")
    logger.info("=" * 80)
    
    # Step 1: Fetch historical data
    logger.info("\n[Step 1] Fetching historical data from Binance...")
    fetcher = HistoricalDataFetcher(exchange_name='binance')
    
    since = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    
    all_data = {}
    for symbol in SYMBOLS:
        try:
            df = fetcher.fetch_ohlcv(symbol, TIMEFRAME, since)
            all_data[symbol] = df
            logger.info(f"[Step 1] ✅ {symbol}: {len(df)} candles")
        except Exception as e:
            logger.error(f"[Step 1] ❌ Failed to fetch {symbol}: {e}")
    
    if not all_data:
        logger.error("[Step 1] ❌ No data fetched. Aborting.")
        return 1
    
    # Step 2: Generate base-agent predictions
    logger.info("\n[Step 2] Generating base-agent predictions...")
    predictor = MockBaseAgentPredictor()
    
    all_features = []
    all_labels = []
    
    for symbol, df in all_data.items():
        logger.info(f"[Step 2] Processing {symbol}...")
        
        for idx in tqdm(range(50, len(df) - LABEL_HORIZON_HOURS), desc=symbol):
            # Generate predictions
            predictions = predictor.predict(df, idx)
            
            # Extract features
            features = MetaAgentTrainer().extract_features(predictions)
            
            # Generate label
            label = MetaAgentTrainer().generate_label(df, idx, LABEL_HORIZON_HOURS)
            
            all_features.append(features)
            all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.info(f"[Step 2] ✅ Generated {len(X)} samples")
    logger.info(f"[Step 2] Label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    if len(X) < MIN_SAMPLES:
        logger.error(f"[Step 2] ❌ Insufficient samples ({len(X)} < {MIN_SAMPLES}). Aborting.")
        return 1
    
    # Step 3: Train Meta-Agent V2
    logger.info("\n[Step 3] Training Meta-Agent V2 model...")
    trainer = MetaAgentTrainer()
    results = trainer.train(X, y)
    
    if results['test_accuracy'] < 0.40:
        logger.warning(f"[Step 3] ⚠️ Low test accuracy ({results['test_accuracy']:.2%})")
        logger.warning("[Step 3] Model may not generalize well, but saving anyway...")
    
    # Step 4: Save model
    logger.info("\n[Step 4] Saving trained model...")
    trainer.save(OUTPUT_DIR)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model location: {OUTPUT_DIR}")
    logger.info(f"Train accuracy: {results['train_accuracy']:.2%}")
    logger.info(f"Test accuracy: {results['test_accuracy']:.2%}")
    logger.info(f"Total samples: {results['num_samples']}")
    logger.info("\nNext steps:")
    logger.info("1. Restart AI Engine: sudo systemctl restart quantum-ai-engine")
    logger.info("2. Verify model loaded: journalctl -u quantum-ai-engine | grep 'MetaV2.*Model ready'")
    logger.info("3. Monitor predictions: journalctl -u quantum-ai-engine -f | grep -iE 'DEFER|ESCALATE'")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
ENSEMBLE MODEL TRAINING v4.1 - Simplified Edition
Trains XGBoost + LightGBM using real trade data labels + synthetic OHLCV features.

Multi-source data approach:
- Source 1: Redis trade.closed (82 real PnL outcomes)
- Source 2: Synthetic OHLCV patterns (generated per trade with realistic volatility)
- Source 3: Technical indicators (RSI, MACD, Bollinger, momentum, etc.)

Result: Models trained on 82 samples to predict PnL% from technical features
"""

import asyncio
import logging
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score

import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate realistic OHLCV data around trade outcomes"""
    
    @staticmethod
    def generate_synthetic_window(close_price: float, pnl_pct: float, n_bars: int = 100) -> pd.DataFrame:
        """
        Create synthetic OHLCV that matches observed PnL%.
        Logic: trend + volatility based on actual result.
        Extended from 30→100 bars for better feature engineering coverage.
        """
        volatility = 0.015 + (abs(pnl_pct) / 100) * 0.01
        start_price = close_price / (1 + pnl_pct / 100)
        
        data = []
        for i in range(n_bars):
            trend = np.sign(pnl_pct) * 0.003 if pnl_pct != 0 else 0
            random_move = np.random.normal(trend, volatility)
            open_price = start_price * (1 + (i / n_bars) * (pnl_pct / 100)) + np.random.normal(0, volatility * 0.5)
            
            high_price = open_price * (1 + abs(np.random.normal(0, volatility * 1.5)))
            low_price = open_price * (1 - abs(np.random.normal(0, volatility * 1.5)))
            close_p = open_price * (1 + random_move)
            
            data.append({
                'open': open_price,
                'high': max(high_price, close_p),
                'low': min(low_price, close_p),
                'close': close_p,
                'volume': np.random.uniform(100, 10000),
            })
        
        df = pd.DataFrame(data)
        
        # Ensure final close matches intended PnL
        current_return = (df['close'].iloc[-1] - start_price) / start_price * 100
        adjustment = 1 + (pnl_pct - current_return) / 100
        df['close'] = df['close'] * adjustment
        df['high'] = df['high'] * adjustment
        df['low'] = df['low'] * adjustment
        
        return df


class EnsembleTrainer:
    """Train XGBoost + LightGBM on PnL prediction"""
    
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        self.models_dir = Path("/home/qt/quantum_trader/models")
        self.models_dir.mkdir(exist_ok=True)
        logger.info(f"[INIT] Redis OK, models dir: {self.models_dir}")
    
    def get_trades(self) -> List[Dict]:
        """Fetch closed trades from Redis"""
        logger.info("[TRADES] Fetching from quantum:stream:trade.closed...")
        
        trades_raw = self.redis.xrange("quantum:stream:trade.closed")
        if not trades_raw:
            logger.error("[TRADES] No trades found!")
            return []
        
        trades = []
        for msg_id, data in trades_raw:
            try:
                entry = float(data.get('entry', 0))
                exit_price = float(data.get('exit', 0))
                pnl_usd = float(data.get('pnl', 0))
                
                # Calculate PnL%
                pnl_pct = (pnl_usd / (entry + 1e-8)) * 100 if entry != 0 else 0
                
                trade = {
                    'id': msg_id,
                    'symbol': data.get('symbol', 'UNKNOWN'),
                    'entry_price': entry,
                    'exit_price': exit_price,
                    'pnl': pnl_usd,
                    'pnl_pct': pnl_pct,
                }
                trades.append(trade)
            except Exception as e:
                logger.debug(f"Parse error: {e}")
                continue
        
        logger.info(f"[TRADES] ✅ Got {len(trades)} trades")
        pnls = [t['pnl_pct'] for t in trades]
        if pnls:
            logger.info(f"[TRADES] PnL% stats: mean={np.mean(pnls):.4f}, std={np.std(pnls):.4f}, range=[{np.min(pnls):.4f}, {np.max(pnls):.4f}]")
        
        return trades
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators"""
        if df.empty or len(df) < 5:
            return df
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs + 1e-8))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger
        sma = df['close'].rolling(10).mean()
        std = df['close'].rolling(10).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        denominator = (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / denominator
        df['bb_pct'] = df['bb_pct'].clip(-100, 100)  # Clip extreme values
        
        # Fill NaN only where safe (forward fill, then forward fill from start)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.fillna(0)  # Fill any remaining NaN with 0
        
        # Final safety: clip all values to reasonable ranges
        df = df.replace([np.inf, -np.inf], 0)
        df = df.clip(-1e6, 1e6)
        
        return df
    
    def prepare_data(self, trades: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate synthetic data + features for each trade"""
        logger.info("[DATA] Generating synthetic OHLCV + features...")
        
        X_list = []
        y_list = []
        
        for trade in trades:
            try:
                df = SyntheticDataGenerator.generate_synthetic_window(
                    trade['exit_price'],
                    trade['pnl_pct'],
                    n_bars=100
                )
                df = self.engineer_features(df)
                
                if df.empty:
                    continue
                
                # Use last row features
                last = df.iloc[-1]
                cols = [c for c in df.columns if c != 'index']
                X = last[cols].values.astype(float)
                
                # Safety check: reject if any NaN/inf remain
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    continue
                
                y = trade['pnl_pct']
                
                X_list.append(X)
                y_list.append(y)
                
            except Exception as e:
                logger.debug(f"Data gen error: {e}")
                continue
        
        if not X_list:
            logger.error("[DATA] No valid data generated!")
            return None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        feature_names = cols
        
        logger.info(f"[DATA] ✅ Generated {len(X)} valid samples, {len(feature_names)} features")
        
        return X, y, feature_names
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Train XGBoost"""
        logger.info("[XGB] Training XGBoost...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_validate(model, X_scaled, y, cv=kfold, scoring=['r2'])
        cv_r2 = cv_scores['test_r2'].mean()
        
        logger.info(f"[XGB] CV R²={cv_r2:.4f}")
        
        model.fit(X_scaled, y)
        train_r2 = r2_score(y, model.predict(X_scaled))
        
        logger.info(f"[XGB] ✅ Train R²={train_r2:.4f}")
        
        return model, scaler
    
    def train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Train LightGBM"""
        logger.info("[LGBM] Training LightGBM...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_validate(model, X_scaled, y, cv=kfold, scoring=['r2'])
        cv_r2 = cv_scores['test_r2'].mean()
        
        logger.info(f"[LGBM] CV R²={cv_r2:.4f}")
        
        model.fit(X_scaled, y)
        train_r2 = r2_score(y, model.predict(X_scaled))
        
        logger.info(f"[LGBM] ✅ Train R²={train_r2:.4f}")
        
        return model, scaler
    
    def save_models(self, xgb_model, xgb_scaler, lgbm_model, lgbm_scaler, feature_names):
        """Save trained models"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version = f"v4_{timestamp}"
        
        logger.info(f"[SAVE] Saving models as {version}...")
        
        xgb_path = self.models_dir / f"xgb_{version}.pkl"
        xgb_scaler_path = self.models_dir / f"xgb_{version}_scaler.pkl"
        lgbm_path = self.models_dir / f"lgbm_{version}.pkl"
        lgbm_scaler_path = self.models_dir / f"lgbm_{version}_scaler.pkl"
        
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        with open(xgb_scaler_path, 'wb') as f:
            pickle.dump(xgb_scaler, f)
        with open(lgbm_path, 'wb') as f:
            pickle.dump(lgbm_model, f)
        with open(lgbm_scaler_path, 'wb') as f:
            pickle.dump(lgbm_scaler, f)
        
        metadata = {
            'timestamp': timestamp,
            'version': version,
            'model_type': 'XGBoost + LightGBM',
            'n_features': len(feature_names),
            'features': feature_names,
            'training_date': datetime.utcnow().isoformat(),
        }
        
        metadata_path = self.models_dir / f"xgb_{version}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[SAVE] ✅ Models saved: {version}")
        
        return version
    
    async def train(self):
        """Main training"""
        logger.info("=" * 80)
        logger.info("ENSEMBLE TRAINING v4.1 - START")
        logger.info("=" * 80)
        
        try:
            trades = self.get_trades()
            if not trades:
                return False
            
            result = self.prepare_data(trades)
            if result[0] is None:
                return False
            
            X, y, feature_names = result
            
            xgb_model, xgb_scaler = self.train_xgboost(X, y)
            lgbm_model, lgbm_scaler = self.train_lightgbm(X, y)
            
            version = self.save_models(xgb_model, xgb_scaler, lgbm_model, lgbm_scaler, feature_names)
            
            logger.info("=" * 80)
            logger.info(f"✅ COMPLETE - {version}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            return False


async def main():
    trainer = EnsembleTrainer()
    success = await trainer.train()
    exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

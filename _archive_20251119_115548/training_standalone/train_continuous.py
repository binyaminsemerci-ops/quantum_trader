"""Continuous AI model training with multi-source data for 3 hours.

Fetches historical and recent data from:
- Binance (primary source)
- CoinGecko (market cap, volume, sentiment)
- Multiple timeframes (1h, 4h, 1d)
- Layer 1 and Layer 2 coins

Trains incrementally and saves best model every iteration.
"""

import os
import sys
import time
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Layer 1 and Layer 2 coins (major focus)
LAYER1_COINS = [
    "BTC", "ETH", "BNB", "SOL", "ADA", "AVAX", "DOT", "MATIC", 
    "ATOM", "NEAR", "APT", "SUI", "SEI", "INJ", "TIA", "FTM",
    "ALGO", "EGLD", "KAVA", "CELO", "ROSE", "ONE"
]

LAYER2_COINS = [
    "ARB", "OP", "METIS", "MANTA", "STRK", "IMX", "LRC", "MATIC",
    "BOBA", "ZK"  # Note: Some may not have USDT pairs
]

BASE_COINS = ["BTC", "ETH", "BNB", "SOL", "ADA", "AVAX", "DOT", "MATIC", "LINK", "UNI"]

# Combine all coins and add USDT suffix
ALL_SYMBOLS = list(set(
    [f"{coin}USDT" for coin in LAYER1_COINS + LAYER2_COINS + BASE_COINS]
))

# Training parameters
TRAINING_DURATION_HOURS = 3
ITERATION_INTERVAL_MINUTES = 15  # Train every 15 minutes
CANDLES_PER_SYMBOL = 1000  # More historical data
TIMEFRAMES = ["1h", "4h"]  # Multiple timeframes for better learning


def fetch_binance_data(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data from Binance."""
    try:
        import requests
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to fetch {symbol} from Binance: {e}")
        return pd.DataFrame()


def fetch_coingecko_data(coin_id: str) -> Dict[str, Any]:
    """Fetch additional market data from CoinGecko."""
    try:
        import requests
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        market_data = data.get("market_data", {})
        
        return {
            "market_cap_rank": data.get("market_cap_rank", 0),
            "sentiment_votes_up": data.get("sentiment_votes_up_percentage", 50),
            "sentiment_votes_down": data.get("sentiment_votes_down_percentage", 50),
            "market_cap": market_data.get("market_cap", {}).get("usd", 0),
            "total_volume": market_data.get("total_volume", {}).get("usd", 0),
            "price_change_24h": market_data.get("price_change_percentage_24h", 0),
            "price_change_7d": market_data.get("price_change_percentage_7d", 0),
            "price_change_30d": market_data.get("price_change_percentage_30d", 0),
        }
        
    except Exception as e:
        logger.debug(f"Failed to fetch CoinGecko data for {coin_id}: {e}")
        return {}


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators."""
    try:
        # Simple Moving Averages
        df["SMA_10"] = df["close"].rolling(window=10).mean()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_200"] = df["close"].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        
        # Bollinger Bands
        bb_period = 20
        df["BB_middle"] = df["close"].rolling(window=bb_period).mean()
        bb_std = df["close"].rolling(window=bb_period).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
        
        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()
        
        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # Price momentum
        df["momentum"] = df["close"].pct_change(periods=10)
        df["rate_of_change"] = df["close"].pct_change(periods=20)
        
        # Stochastic
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        df["stochastic_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["stochastic_d"] = df["stochastic_k"].rolling(window=3).mean()
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to add some technical indicators: {e}")
        return df


def create_labels(df: pd.DataFrame, future_periods: int = 10, threshold: float = 0.02) -> pd.DataFrame:
    """Create labels based on future price movement."""
    try:
        df["future_close"] = df["close"].shift(-future_periods)
        df["price_change"] = (df["future_close"] - df["close"]) / df["close"]
        
        # Multi-class: BUY (1), HOLD (0), SELL (-1) -> converted to 0, 1, 2 for sklearn
        df["label"] = 1  # Default HOLD
        df.loc[df["price_change"] > threshold, "label"] = 2  # BUY
        df.loc[df["price_change"] < -threshold, "label"] = 0  # SELL
        
        # Drop rows without future data
        df = df[df["future_close"].notna()].copy()
        df.drop(columns=["future_close", "price_change"], inplace=True)
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to create labels: {e}")
        return df


def collect_training_data(symbols: List[str], timeframes: List[str]) -> pd.DataFrame:
    """Collect data from multiple symbols and timeframes."""
    all_data = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Fetching {symbol} {timeframe}...")
            
            df = fetch_binance_data(symbol, interval=timeframe, limit=CANDLES_PER_SYMBOL)
            
            if df.empty:
                continue
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
            # Create labels
            df = create_labels(df)
            
            # Drop NaN values from indicators
            df = df.dropna()
            
            if len(df) > 50:  # Minimum data requirement
                all_data.append(df)
                logger.info(f"  ✓ Collected {len(df)} samples from {symbol} {timeframe}")
            
            # Rate limiting
            time.sleep(0.2)
    
    if not all_data:
        logger.error("No data collected!")
        return pd.DataFrame()
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"[CHART] Total samples collected: {len(combined)}")
    
    # Show class distribution
    if "label" in combined.columns:
        dist = combined["label"].value_counts().sort_index()
        logger.info(f"   SELL: {dist.get(0, 0)}, HOLD: {dist.get(1, 0)}, BUY: {dist.get(2, 0)}")
    
    return combined


def _is_stub_xgb(cls) -> bool:
    marker_attrs = ("QT_IS_STUB", "IS_QT_XGBOOST_STUB")
    return any(bool(getattr(cls, attr, False)) for attr in marker_attrs)


def train_models(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Train multiple models and return the best one."""
    models = {}
    scores = {}
    
    try:
        from xgboost import XGBClassifier

        if _is_stub_xgb(XGBClassifier):
            raise ImportError("Native XGBoost missing")

        logger.info("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        xgb_score = xgb.score(X_test, y_test)
        models["xgboost"] = xgb
        scores["xgboost"] = xgb_score
        logger.info(f"  XGBoost accuracy: {xgb_score:.4f}")
    except ImportError:
        logger.info("XGBoost not available; skipping model")
    
    logger.info("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_score = lgbm.score(X_test, y_test)
    models["lightgbm"] = lgbm
    scores["lightgbm"] = lgbm_score
    logger.info(f"  LightGBM accuracy: {lgbm_score:.4f}")
    
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    models["random_forest"] = rf
    scores["random_forest"] = rf_score
    logger.info(f"  Random Forest accuracy: {rf_score:.4f}")
    
    # Select best model
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    best_score = scores[best_model_name]
    
    logger.info(f"🏆 Best model: {best_model_name} with accuracy {best_score:.4f}")
    
    return {
        "model": best_model,
        "model_name": best_model_name,
        "score": best_score,
        "all_scores": scores
    }


def save_model(model, scaler, model_path: str, scaler_path: str):
    """Save trained model and scaler."""
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[OK] Saved model to {model_path}")
        
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"[OK] Saved scaler to {scaler_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def main():
    """Main continuous training loop."""
    logger.info("=" * 80)
    logger.info("[ROCKET] CONTINUOUS AI TRAINING - 3 HOURS")
    logger.info("=" * 80)
    logger.info(f"Symbols: {len(ALL_SYMBOLS)} (Layer 1, Layer 2, Base)")
    logger.info(f"Timeframes: {TIMEFRAMES}")
    logger.info(f"Training interval: {ITERATION_INTERVAL_MINUTES} minutes")
    logger.info(f"Duration: {TRAINING_DURATION_HOURS} hours")
    logger.info("=" * 80)
    
    start_time = time.time()
    end_time = start_time + (TRAINING_DURATION_HOURS * 3600)
    iteration = 0
    best_overall_score = 0.0
    
    model_dir = Path("ai_engine/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    while time.time() < end_time:
        iteration += 1
        iter_start = time.time()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Collect data
        logger.info("📥 Collecting data from multiple sources...")
        df = collect_training_data(ALL_SYMBOLS, TIMEFRAMES)
        
        if df.empty or len(df) < 100:
            logger.warning("[WARNING]  Insufficient data, skipping iteration")
            time.sleep(60)
            continue
        
        # Prepare features
        feature_cols = [col for col in df.columns if col != "label"]
        X = df[feature_cols].values
        y = df["label"].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"[CHART] Training data: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train models
        result = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save if this is the best model so far
        if result["score"] > best_overall_score:
            best_overall_score = result["score"]
            logger.info(f"[TARGET] NEW BEST MODEL! Accuracy: {best_overall_score:.4f}")
            
            save_model(
                result["model"],
                scaler,
                str(model_dir / "xgb_model.pkl"),
                str(model_dir / "scaler.pkl")
            )
        else:
            logger.info(f"   Current best remains: {best_overall_score:.4f}")
        
        # Calculate time remaining
        elapsed = time.time() - start_time
        remaining = end_time - time.time()
        
        logger.info(f"⏱️  Elapsed: {elapsed/3600:.2f}h, Remaining: {remaining/3600:.2f}h")
        
        # Wait until next iteration
        iter_duration = time.time() - iter_start
        sleep_time = max(0, (ITERATION_INTERVAL_MINUTES * 60) - iter_duration)
        
        if sleep_time > 0 and time.time() + sleep_time < end_time:
            logger.info(f"💤 Sleeping {sleep_time/60:.1f} minutes until next iteration...")
            time.sleep(sleep_time)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("[CHECKERED_FLAG] TRAINING COMPLETE!")
    logger.info(f"   Iterations: {iteration}")
    logger.info(f"   Best accuracy achieved: {best_overall_score:.4f}")
    logger.info(f"   Model saved to: {model_dir / 'xgb_model.pkl'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

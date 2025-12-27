#!/usr/bin/env python3
"""
Standalone training script that doesn't depend on backend imports.
Uses direct API calls to get live data and trains the AI model.
"""

import asyncio
import aiohttp
import pickle
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ai_engine", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class SimpleScaler:
    """Simple scaler for normalization"""

    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def _is_stub_xgb(cls) -> bool:
    """Return True when the imported XGBoost class is the lightweight stub."""

    marker_attrs = ("QT_IS_STUB", "IS_QT_XGBOOST_STUB")
    return any(bool(getattr(cls, attr, False)) for attr in marker_attrs)


async def fetch_binance_data(symbol: str, interval: str = "1h", limit: int = 500):
    """Fetch OHLCV data from Binance public API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Convert to OHLCV format
                    candles = []
                    for item in data:
                        candle = {
                            "timestamp": int(item[0]),
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": float(item[5]),
                        }
                        candles.append(candle)

                    return candles
                else:
                    logger.error(f"Failed to fetch {symbol}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return []


async def fetch_coingecko_sentiment(symbol: str):
    """Get sentiment score from CoinGecko trending data"""
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if symbol is in trending (simplified sentiment)
                    symbol_clean = symbol.replace("USDT", "").replace("BTC", "").lower()

                    for coin in data.get("coins", []):
                        if (
                            coin.get("item", {}).get("symbol", "").lower()
                            == symbol_clean
                        ):
                            return 0.7  # Positive sentiment if trending

                    return 0.5  # Neutral sentiment
                else:
                    return 0.5
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        return 0.5


def add_technical_indicators(df):
    """Add technical indicators to DataFrame"""
    # Simple moving averages
    df["SMA_10"] = df["close"].rolling(window=10).mean()
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["EMA_10"] = df["close"].ewm(span=10).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df["close"].rolling(window=20).mean()
    rolling_std = df["close"].rolling(window=20).std()
    df["BB_upper"] = rolling_mean + (rolling_std * 2)
    df["BB_lower"] = rolling_mean - (rolling_std * 2)

    # MACD
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Price change
    df["price_change_1h"] = df["close"].pct_change(1)
    df["price_change_24h"] = df["close"].pct_change(24)

    # Volume features
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    return df


def create_labels(df, future_periods=4):
    """Create training labels based on future price movements"""
    # Calculate future price change
    future_close = df["close"].shift(-future_periods)
    price_change = (future_close - df["close"]) / df["close"]

    # Create labels: 2 for significant upward movement, 0 for downward, 1 for sideways
    # Then we'll filter to only use 0 and 2 (which become 0 and 1)
    labels = np.where(price_change > 0.02, 2, np.where(price_change < -0.02, 0, 1))

    return labels


async def collect_training_data():
    """Collect training data from multiple symbols"""
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "MATICUSDT",
        "LINKUSDT",
        "UNIUSDT",
    ]

    all_features = []
    all_labels = []

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}")

        # Get OHLCV data
        candles = await fetch_binance_data(symbol, limit=500)
        if not candles:
            continue

        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Get sentiment
        sentiment = await fetch_coingecko_sentiment(symbol)
        df["sentiment"] = sentiment

        # Add technical indicators
        df = add_technical_indicators(df)

        # Create labels
        labels = create_labels(df)

        # Select features
        feature_columns = [
            "SMA_10",
            "SMA_20",
            "EMA_10",
            "RSI",
            "BB_upper",
            "BB_lower",
            "MACD",
            "MACD_signal",
            "price_change_1h",
            "price_change_24h",
            "volume_ratio",
            "sentiment",
        ]

        # Filter valid rows (drop NaN)
        valid_mask = df[feature_columns].notna().all(axis=1)
        valid_mask = valid_mask & (
            labels != 1
        )  # Only use non-neutral labels (0=down, 2=up)

        if valid_mask.sum() > 10:  # Need at least some valid samples
            features = df.loc[valid_mask, feature_columns].values
            valid_labels = labels[valid_mask]

            # Convert labels to 0,1 format (0=down, 1=up)
            valid_labels = np.where(valid_labels == 2, 1, 0)

            all_features.append(features)
            all_labels.extend(valid_labels)

        # Add delay to be respectful to APIs
        await asyncio.sleep(0.5)

    if not all_features:
        raise ValueError("No training data collected")

    X = np.vstack(all_features)
    y = np.array(all_labels)

    logger.info(f"Collected {len(X)} training samples")
    return X, y


async def train_model():
    """Train and save the AI model"""
    logger.info("Starting AI model training with live data...")

    # Collect training data
    X, y = await collect_training_data()

    # Split into train/test so metadata can report validation accuracy
    stratify_labels = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )

    # Prepare scaler using training data only and reuse for validation
    scaler = SimpleScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    try:
        from xgboost import XGBClassifier

        if _is_stub_xgb(XGBClassifier):
            raise ImportError("Native XGBoost not available")

        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        logger.info("Using XGBoost classifier")
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        logger.info("Using Random Forest classifier (XGBoost not available)")

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate on the holdout split for quick sanity metrics
    validation_accuracy = float(model.score(X_test_scaled, y_test))
    logger.info("Validation accuracy: %.3f", validation_accuracy)

    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "samples": len(X),
        "features": X.shape[1],
        "model_type": type(model).__name__,
        "accuracy": validation_accuracy,
    }

    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    logger.info("AI model training completed successfully!")
    return model, scaler


if __name__ == "__main__":
    asyncio.run(train_model())

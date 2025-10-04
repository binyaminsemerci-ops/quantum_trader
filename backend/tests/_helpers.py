"""
Test helper functions and stubs for Quantum Trader backend tests.

This module provides common test utilities, mocks, and stubs used across
the test suite to avoid code duplication and ensure consistent test behavior.
"""

from typing import List, Dict, Any


class TrainAndSaveStub:
    """
    Stub module for training and saving ML models in tests.

    This replaces the actual train_and_save module to avoid expensive
    computation and external dependencies during testing.
    """

    # Default model directory (can be overridden in tests)
    MODEL_DIR = "/tmp/test_models"

    def train_and_save(self, symbols: List[str], samples: int = 600):
        """
        Mock training and saving function.

        Args:
            symbols: List of trading symbols to train on
            samples: Number of samples to use for training

        Returns:
            dict: Mock training results
        """
        import json
        import pickle
        from pathlib import Path

        # Ensure model directory exists
        model_dir = Path(self.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create mock model files
        model_path = model_dir / "xgb_model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        metadata_path = model_dir / "metadata.json"

        # Save mock pickle files
        mock_model = {"type": "xgboost", "features": ["price", "volume"]}
        mock_scaler = {"mean": [100.0, 1000.0], "std": [10.0, 100.0]}

        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(mock_scaler, f)

        # Save metadata
        metadata = {
            "status": "completed",
            "model_path": str(model_path),
            "accuracy": 0.85,
            "features_used": ["price", "volume", "rsi"],
            "training_samples": samples,
            "symbols": symbols,
            "trained_at": "2025-10-04T13:50:00Z"
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata


# Create global instances for easy importing
train_and_save_stub = TrainAndSaveStub()


class ExternalDataStub:
    """
    Stub class for external data sources used in tests.

    This replaces actual API calls to external services like Binance
    with predictable test data.
    """

    async def binance_ohlcv(self, symbol: str, limit: int = 600) -> List[Dict[str, Any]]:
        """
        Mock OHLCV data from Binance API.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            limit: Number of candles to return

        Returns:
            List of OHLCV candle dictionaries
        """
        candles = []
        base_price = 100.0

        for i in range(limit):
            # Generate predictable price movement
            price_change = (i % 10 - 5) * 0.01  # Small oscillations
            open_price = base_price + price_change
            close_price = open_price + ((i % 3 - 1) * 0.005)  # Random-ish close
            high_price = max(open_price, close_price) + 0.002
            low_price = min(open_price, close_price) - 0.002
            volume = 1000 + (i % 100)  # Varying volume

            candle = {
                "timestamp": 1609459200 + (i * 60),  # Start from 2021-01-01, 1min intervals
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            }
            candles.append(candle)

        return candles

    async def sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Mock sentiment data for testing.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with sentiment metrics
        """
        return {
            "symbol": symbol,
            "sentiment_score": 0.65,
            "social_mentions": 150,
            "news_sentiment": "positive",
            "fear_greed_index": 45
        }

    async def twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Mock Twitter sentiment data for testing.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with Twitter sentiment metrics
        """
        return {
            "symbol": symbol,
            "tweets": [
                {"text": f"Bullish on {symbol}!", "sentiment": 0.8},
                {"text": f"{symbol} looking strong", "sentiment": 0.6},
                {"text": f"Not sure about {symbol}", "sentiment": 0.1}
            ],
            "overall_sentiment": 0.5,
            "tweet_count": 25
        }



# Create a global instance for easy importing
external_data_stub = ExternalDataStub()

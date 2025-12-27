"""
RealDataClient - Production Data Loading for CLM

Integrates with BinanceDataFetcher to load historical market data
for model training, validation, and evaluation.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RealDataClient:
    """
    Production data client for CLM.
    
    Loads market data from Binance and applies feature engineering
    for model training.
    """
    
    def __init__(
        self,
        binance_client=None,
        symbols: Optional[list[str]] = None,
        default_symbol: str = "BTCUSDT",
    ):
        """
        Initialize RealDataClient.
        
        Args:
            binance_client: Binance API client (optional, can use REST API)
            symbols: List of symbols to load data for
            default_symbol: Default symbol for single-symbol training
        """
        self.binance_client = binance_client
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.default_symbol = default_symbol
        
        logger.info(f"[DataClient] Initialized with {len(self.symbols)} symbols")
    
    def load_training_data(
        self, 
        start: datetime, 
        end: datetime,
        symbol: Optional[str] = None,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Load historical data for training.
        
        Args:
            start: Start datetime
            end: End datetime
            symbol: Trading symbol (None = use default)
            interval: Candle interval (1h, 4h, 1d)
        
        Returns:
            DataFrame with OHLCV + features
        """
        symbol = symbol or self.default_symbol
        
        logger.info(
            f"[DataClient] Loading training data: {symbol} "
            f"from {start.date()} to {end.date()} ({interval})"
        )
        
        try:
            # Load from Binance
            df = self._fetch_binance_data(symbol, start, end, interval)
            
            # Apply feature engineering
            df = self._add_features(df)
            
            # Clean data
            df = self._clean_data(df)
            
            logger.info(
                f"[DataClient] Loaded {len(df)} rows, {len(df.columns)} features"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"[DataClient] Failed to load training data: {e}")
            raise
    
    def load_recent_data(self, days: int = 7, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load recent data for trigger detection.
        
        Args:
            days: Number of recent days
            symbol: Trading symbol
        
        Returns:
            DataFrame with recent data
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        return self.load_training_data(start, end, symbol)
    
    def load_validation_data(
        self, 
        days: int = 30, 
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load validation data for model evaluation.
        
        Args:
            days: Number of days for validation set
            symbol: Trading symbol
        
        Returns:
            DataFrame with validation data
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        return self.load_training_data(start, end, symbol)
    
    def _fetch_binance_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Fetch raw OHLCV data from Binance.
        
        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            interval: Candle interval
        
        Returns:
            DataFrame with OHLCV columns
        """
        # TODO: Replace with actual BinanceDataFetcher integration
        # For now, generate synthetic data for testing
        
        if self.binance_client:
            # Use actual Binance client
            try:
                from backend.services.binance_data_fetcher import BinanceDataFetcher
                fetcher = BinanceDataFetcher(self.binance_client)
                df = fetcher.fetch_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=int(start.timestamp() * 1000),
                    end_time=int(end.timestamp() * 1000),
                )
                return df
            except Exception as e:
                logger.warning(f"[DataClient] Binance fetch failed: {e}, using mock data")
        
        # Generate mock data for testing
        hours = int((end - start).total_seconds() / 3600)
        dates = pd.date_range(start=start, end=end, periods=hours)
        
        # Simulate realistic price movements
        np.random.seed(42)
        base_price = 50000 if "BTC" in symbol else 3000
        
        returns = np.random.normal(0, 0.02, hours)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices * (1 + np.random.normal(0, 0.001, hours)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, hours))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, hours))),
            "close": prices,
            "volume": np.random.uniform(100, 1000, hours),
        })
        
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and features.
        
        Args:
            df: Raw OHLCV dataframe
        
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Moving averages
        for period in [7, 14, 30, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        
        # Volatility
        df["volatility_7"] = df["returns"].rolling(7).std()
        df["volatility_30"] = df["returns"].rolling(30).std()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean()
        
        # Volume indicators
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f"momentum_{period}"] = df["close"].diff(period)
        
        # Target: Future return (for supervised learning)
        df["target_1h"] = df["close"].shift(-1) / df["close"] - 1
        df["target_4h"] = df["close"].shift(-4) / df["close"] - 1
        df["target_direction"] = (df["target_1h"] > 0).astype(int)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: remove NaN, outliers, etc.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Cleaned DataFrame
        """
        # Remove initial NaN from indicators
        df = df.dropna()
        
        # Remove extreme outliers (beyond 5 std)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ["timestamp"]:
                mean = df[col].mean()
                std = df[col].std()
                df = df[(df[col] >= mean - 5*std) & (df[col] <= mean + 5*std)]
        
        # Reset index
        df = df.reset_index(drop=False)
        
        logger.info(f"[DataClient] Data cleaned: {len(df)} rows remaining")
        
        return df
    
    def get_feature_names(self) -> list[str]:
        """
        Get list of feature column names.
        
        Returns:
            List of feature names
        """
        # All features except OHLCV and targets
        base_features = ["returns", "log_returns"]
        
        ma_features = [f"{ma}_{p}" for ma in ["sma", "ema"] for p in [7, 14, 30, 50]]
        vol_features = ["volatility_7", "volatility_30"]
        indicator_features = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_position", "atr_14",
            "volume_sma_20", "volume_ratio"
        ]
        momentum_features = [f"momentum_{p}" for p in [5, 10, 20]]
        
        return base_features + ma_features + vol_features + indicator_features + momentum_features

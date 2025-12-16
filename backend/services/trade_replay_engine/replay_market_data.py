"""
Replay Market Data Source - Provides historical candle data for replay
"""

import logging
from datetime import datetime, timedelta
from typing import Protocol, Iterator
import pandas as pd

from .replay_config import ReplayConfig

logger = logging.getLogger(__name__)


class MarketDataClient(Protocol):
    """Protocol for market data providers"""
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        ...


class ReplayMarketDataSource:
    """
    Loads and provides historical market data for replay sessions.
    
    Handles data loading, validation, and time-stepped iteration.
    """
    
    def __init__(self, market_data_client: MarketDataClient):
        """
        Args:
            market_data_client: Client for fetching historical data
        """
        self.client = market_data_client
        self.logger = logging.getLogger(f"{__name__}.ReplayMarketDataSource")
    
    def load(self, config: ReplayConfig) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for all symbols/timeframe between start and end.
        
        Args:
            config: Replay configuration
        
        Returns:
            dict: symbol -> DataFrame with OHLCV data
        
        Raises:
            ValueError: If data loading fails
        """
        self.logger.info(
            f"Loading data for {len(config.symbols)} symbols "
            f"from {config.start} to {config.end}"
        )
        
        data = {}
        
        for symbol in config.symbols:
            try:
                df = self.client.get_historical_candles(
                    symbol=symbol,
                    timeframe=config.timeframe,
                    start=config.start,
                    end=config.end
                )
                
                if df is None or df.empty:
                    self.logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # Validate data
                if not self._validate_data(df, symbol):
                    self.logger.warning(f"Invalid data for {symbol}, skipping")
                    continue
                
                data[symbol] = df
                self.logger.info(
                    f"Loaded {len(df)} candles for {symbol} "
                    f"({df.index[0]} to {df.index[-1]})"
                )
            
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
                continue
        
        if not data:
            raise ValueError("Failed to load data for any symbol")
        
        self.logger.info(f"Successfully loaded data for {len(data)} symbols")
        return data
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate OHLCV data"""
        required_cols = ["open", "high", "low", "close", "volume"]
        
        # Check columns
        for col in required_cols:
            if col not in df.columns:
                self.logger.error(f"{symbol}: Missing column {col}")
                return False
        
        # Check for NaN
        if df[required_cols].isnull().any().any():
            self.logger.warning(f"{symbol}: Contains NaN values")
            # Don't fail, just warn
        
        # Check OHLC logic
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"{symbol}: Invalid OHLC logic in {invalid_ohlc.sum()} candles")
        
        # Check for negative prices
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            self.logger.error(f"{symbol}: Contains non-positive prices")
            return False
        
        return True
    
    def iter_time_steps(self, data: dict[str, pd.DataFrame]) -> Iterator[datetime]:
        """
        Returns a generator over candle timestamps in correct order.
        
        Yields unique timestamps across all symbols in chronological order.
        
        Args:
            data: dict of symbol -> DataFrame
        
        Yields:
            datetime: Next timestamp
        """
        # Collect all timestamps
        all_timestamps = set()
        
        for symbol, df in data.items():
            all_timestamps.update(df.index.tolist())
        
        # Sort and yield
        sorted_timestamps = sorted(all_timestamps)
        
        self.logger.info(
            f"Prepared {len(sorted_timestamps)} time steps "
            f"from {sorted_timestamps[0]} to {sorted_timestamps[-1]}"
        )
        
        for ts in sorted_timestamps:
            yield ts
    
    def get_candles_at_time(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> dict[str, dict]:
        """
        Get candle data for all symbols at a specific timestamp.
        
        Args:
            data: dict of symbol -> DataFrame
            timestamp: Timestamp to query
        
        Returns:
            dict: symbol -> candle dict with OHLCV
        """
        candles = {}
        
        for symbol, df in data.items():
            if timestamp in df.index:
                row = df.loc[timestamp]
                candles[symbol] = {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
        
        return candles
    
    def get_price_snapshot(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> dict[str, float]:
        """
        Get close prices for all symbols at a specific timestamp.
        
        Args:
            data: dict of symbol -> DataFrame
            timestamp: Timestamp to query
        
        Returns:
            dict: symbol -> close price
        """
        prices = {}
        
        for symbol, df in data.items():
            if timestamp in df.index:
                prices[symbol] = float(df.loc[timestamp]["close"])
        
        return prices

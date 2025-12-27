"""
Binance market data client implementation.

Fetches historical OHLCV data from Binance Futures API.
"""

import logging
from datetime import datetime
from typing import Optional
import pandas as pd
from binance.client import Client

from .repositories import MarketDataClient

logger = logging.getLogger(__name__)


class BinanceMarketDataClient(MarketDataClient):
    """
    Fetches historical OHLCV data from Binance Futures.
    
    Integrates with existing Quantum Trader Binance client.
    """
    
    # Interval mapping
    INTERVAL_MAP = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR,
        "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        "3d": Client.KLINE_INTERVAL_3DAY,
        "1w": Client.KLINE_INTERVAL_1WEEK,
        "1M": Client.KLINE_INTERVAL_1MONTH,
    }
    
    def __init__(self, binance_client: Client):
        """
        Initialize market data client.
        
        Args:
            binance_client: Initialized Binance client instance
        """
        self.client = binance_client
        self._cache = {}  # Simple in-memory cache
    
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Interval (e.g., "15m", "1h", "4h")
            start: Start timestamp
            end: End timestamp
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Check cache
        cache_key = f"{symbol}:{timeframe}:{start.isoformat()}:{end.isoformat()}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {symbol} {timeframe}")
            return self._cache[cache_key].copy()
        
        try:
            # Map timeframe to Binance interval
            interval = self.INTERVAL_MAP.get(timeframe)
            if not interval:
                logger.warning(f"Unknown timeframe {timeframe}, defaulting to 15m")
                interval = Client.KLINE_INTERVAL_15MINUTE
            
            # Convert timestamps to milliseconds
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            
            logger.info(
                f"Fetching {symbol} {timeframe} data: "
                f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}"
            )
            
            # Fetch klines from Binance Futures
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=1500  # Max per request
            )
            
            if not klines:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Parse types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Keep only required columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Cache result (limit cache size to 100 entries)
            if len(self._cache) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = df.copy()
            
            logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe} data: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def clear_cache(self):
        """Clear the internal cache"""
        self._cache.clear()
        logger.info("Market data cache cleared")

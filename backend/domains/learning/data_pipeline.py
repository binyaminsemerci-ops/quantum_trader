"""
Data Pipeline - Historical data fetching and feature engineering for ML models.

Provides:
- HistoricalDataFetcher: Fetch OHLCV data from Binance + database
- FeatureEngineer: Compute technical indicators (RSI, MACD, BB, ATR, etc.)
- Label generation for supervised learning

Compatible with XGBoost, LightGBM, N-HiTS, PatchTST.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    
    # Volume indicators
    volume_sma_period: int = 20
    
    # Volatility
    volatility_window: int = 24
    
    # Price action
    lookback_candles: int = 100
    
    # Label generation
    label_horizon: int = 12  # How many candles ahead to predict
    label_type: str = "future_return"  # or "direction", "regime"
    
    def to_dict(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "atr_period": self.atr_period,
            "ema_periods": self.ema_periods,
            "volume_sma_period": self.volume_sma_period,
            "volatility_window": self.volatility_window,
            "lookback_candles": self.lookback_candles,
            "label_horizon": self.label_horizon,
            "label_type": self.label_type,
        }


# ============================================================================
# Historical Data Fetcher
# ============================================================================

class HistoricalDataFetcher:
    """
    Fetch historical OHLCV data from Binance API and/or database.
    
    Supports multiple timeframes and symbols.
    Handles rate limiting and retries.
    """
    
    BINANCE_BASE_URL = "https://fapi.binance.com"
    BINANCE_TESTNET_URL = "https://testnet.binancefuture.com"
    
    TIMEFRAME_TO_MINUTES = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
    }
    
    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        use_testnet: bool = False,
        max_concurrent: int = 5,
    ):
        self.db = db_session
        self.base_url = self.BINANCE_TESTNET_URL if use_testnet else self.BINANCE_BASE_URL
        self.max_concurrent = max_concurrent
        
    async def fetch_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "5m",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            start_date: Start of data range
            end_date: End of data range
            timeframe: Candle interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        logger.info(
            f"Fetching historical data: {len(symbols)} symbols, "
            f"{start_date.date()} to {end_date.date()}, timeframe={timeframe}"
        )
        
        # Try database first if available
        if self.db:
            try:
                df = await self._fetch_from_database(symbols, start_date, end_date, timeframe)
                if not df.empty:
                    logger.info(f"Loaded {len(df)} rows from database")
                    return df
            except Exception as e:
                logger.warning(f"Database fetch failed: {e}, falling back to API")
        
        # Fetch from Binance API
        df = await self._fetch_from_binance(symbols, start_date, end_date, timeframe)
        logger.info(f"Fetched {len(df)} rows from Binance API")
        
        # Store in database for future use
        if self.db and not df.empty:
            try:
                await self._store_to_database(df, timeframe)
                logger.info("Stored data to database")
            except Exception as e:
                logger.warning(f"Failed to store to database: {e}")
        
        return df
    
    async def _fetch_from_database(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch data from local database."""
        if not self.db:
            return pd.DataFrame()
        
        # Query historical klines table
        query = text("""
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume
            FROM historical_klines
            WHERE 
                symbol = ANY(:symbols)
                AND timeframe = :timeframe
                AND timestamp >= :start_date
                AND timestamp <= :end_date
            ORDER BY symbol, timestamp
        """)
        
        result = await self.db.execute(
            query,
            {
                "symbols": symbols,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    
    async def _fetch_from_binance(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch data from Binance API with rate limiting."""
        
        # Fetch symbols in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_symbol(symbol: str) -> pd.DataFrame:
            async with semaphore:
                return await self._fetch_single_symbol(symbol, start_date, end_date, timeframe)
        
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        dfs = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {symbol}: {result}")
                continue
            if not result.empty:
                dfs.append(result)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    async def _fetch_single_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch klines for a single symbol."""
        
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ms
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            while current_start < end_ms:
                try:
                    response = await client.get(
                        f"{self.base_url}/fapi/v1/klines",
                        params={
                            "symbol": symbol,
                            "interval": timeframe,
                            "startTime": current_start,
                            "endTime": end_ms,
                            "limit": 1500,  # Binance max
                        }
                    )
                    response.raise_for_status()
                    klines = response.json()
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    
                    # Move start time forward
                    current_start = klines[-1][0] + 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error fetching {symbol} klines: {e}")
                    break
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Keep only needed columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["symbol"] = symbol
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    async def _store_to_database(self, df: pd.DataFrame, timeframe: str) -> None:
        """Store fetched data to database."""
        if not self.db or df.empty:
            return
        
        # Insert or update (upsert)
        for _, row in df.iterrows():
            query = text("""
                INSERT INTO historical_klines 
                (timestamp, symbol, timeframe, open, high, low, close, volume)
                VALUES (:timestamp, :symbol, :timeframe, :open, :high, :low, :close, :volume)
                ON CONFLICT (symbol, timeframe, timestamp) 
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """)
            
            await self.db.execute(query, {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "timeframe": timeframe,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })
        
        await self.db.commit()


# ============================================================================
# Feature Engineer
# ============================================================================

class FeatureEngineer:
    """
    Compute technical indicators and features for ML models.
    
    DEPRECATED: This class now wraps UnifiedFeatureEngineer for backward compatibility.
    New code should use: from backend.shared.unified_features import get_feature_engineer
    
    Generates 50+ features including:
    - Momentum indicators (RSI, MACD, Stochastic)
    - Trend indicators (EMA, SMA, ADX)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators
    - Price patterns
    - Market microstructure features
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        # Use unified feature engineer internally
        from backend.shared.unified_features import get_feature_engineer
        self._unified_engineer = get_feature_engineer()
        logger.info("✅ FeatureEngineer initialized with UnifiedFeatureEngineer (50+ features)")
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add technical indicator features to OHLCV data.
        
        Args:
            df: DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
            symbol: If provided, process only this symbol
            
        Returns:
            DataFrame with 50+ additional feature columns
        """
        if df.empty:
            return df
        
        # Process each symbol separately
        if "symbol" in df.columns and symbol is None:
            symbols = df["symbol"].unique()
            dfs = []
            for sym in symbols:
                sym_df = df[df["symbol"] == sym].copy()
                # Use unified feature engineer
                try:
                    sym_df = self._unified_engineer.compute_features(sym_df)
                    logger.debug(f"✅ Computed {len(sym_df.columns)} unified features for {sym}")
                except Exception as e:
                    logger.error(f"❌ Unified feature computation failed for {sym}: {e}")
                    # Fallback to legacy method
                    sym_df = self._compute_features(sym_df)
                dfs.append(sym_df)
            return pd.concat(dfs, ignore_index=True)
        else:
            # Use unified feature engineer
            try:
                result = self._unified_engineer.compute_features(df.copy())
                logger.debug(f"✅ Computed {len(result.columns)} unified features")
                return result
            except Exception as e:
                logger.error(f"❌ Unified feature computation failed: {e}, falling back to legacy")
                return self._compute_features(df.copy())
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a single symbol."""
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Price features
        df = self._add_price_features(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Trend indicators
        df = self._add_trend_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Pattern features
        df = self._add_pattern_features(df)
        
        # Microstructure features
        df = self._add_microstructure_features(df)
        
        # Drop NaN rows (from indicator warmup)
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with NaN values after feature engineering")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-derived features."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        df["body_size"] = abs(df["close"] - df["open"]) / df["close"]
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, Stochastic, ROC."""
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Stochastic Oscillator
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        
        # Rate of Change
        df["roc"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMAs, SMAs, ADX."""
        
        # EMAs
        for period in self.config.ema_periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"ema_{period}_dist"] = (df["close"] - df[f"ema_{period}"]) / df["close"]
        
        # SMA
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        
        # ADX (Average Directional Index)
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Bands, Volatility."""
        
        # ATR (already computed in ADX)
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=self.config.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        
        # Bollinger Bands
        sma = df["close"].rolling(window=self.config.bb_period).mean()
        std = df["close"].rolling(window=self.config.bb_period).std()
        df["bb_upper"] = sma + (std * self.config.bb_std)
        df["bb_lower"] = sma - (std * self.config.bb_std)
        df["bb_middle"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # Historical Volatility
        df["volatility"] = df["returns"].rolling(window=self.config.volatility_window).std()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        
        df["volume_sma"] = df["volume"].rolling(window=self.config.volume_sma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        
        # Volume-weighted average price
        df["vwap"] = (df["close"] * df["volume"]).rolling(window=20).sum() / df["volume"].rolling(window=20).sum()
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick patterns and price patterns."""
        
        # Doji
        df["is_doji"] = (abs(df["close"] - df["open"]) / df["price_range"] < 0.1).astype(int)
        
        # Hammer / Hanging Man
        body = abs(df["close"] - df["open"])
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        df["is_hammer"] = ((lower_shadow > 2 * body) & (df["upper_wick"] < body)).astype(int)
        
        # Engulfing patterns
        df["bullish_engulfing"] = (
            (df["close"] > df["open"]) &
            (df["close"].shift(1) < df["open"].shift(1)) &
            (df["close"] > df["open"].shift(1)) &
            (df["open"] < df["close"].shift(1))
        ).astype(int)
        
        df["bearish_engulfing"] = (
            (df["close"] < df["open"]) &
            (df["close"].shift(1) > df["open"].shift(1)) &
            (df["close"] < df["open"].shift(1)) &
            (df["open"] > df["close"].shift(1))
        ).astype(int)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure and statistical features."""
        
        # Rolling statistics
        df["returns_mean_5"] = df["returns"].rolling(window=5).mean()
        df["returns_std_5"] = df["returns"].rolling(window=5).std()
        df["returns_skew_10"] = df["returns"].rolling(window=10).skew()
        df["returns_kurt_10"] = df["returns"].rolling(window=10).kurt()
        
        # Autocorrelation
        df["returns_autocorr_1"] = df["returns"].rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        
        # Price momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        return df
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        label_type: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate labels for supervised learning.
        
        Args:
            df: DataFrame with OHLCV and features
            label_type: "future_return", "direction", or "regime"
            horizon: How many candles ahead to predict
            
        Returns:
            DataFrame with additional "label" column
        """
        label_type = label_type or self.config.label_type
        horizon = horizon or self.config.label_horizon
        
        df = df.copy()
        
        if label_type == "future_return":
            # Predict future return (regression)
            df["label"] = df["close"].pct_change(horizon).shift(-horizon)
            
        elif label_type == "direction":
            # Predict direction: 1 (up), 0 (down)
            future_return = df["close"].pct_change(horizon).shift(-horizon)
            df["label"] = (future_return > 0).astype(int)
            
        elif label_type == "regime":
            # Predict market regime: 0 (bearish), 1 (neutral), 2 (bullish)
            future_return = df["close"].pct_change(horizon).shift(-horizon)
            df["label"] = pd.cut(
                future_return,
                bins=[-np.inf, -0.02, 0.02, np.inf],
                labels=[0, 1, 2]
            ).astype(int)
        
        else:
            raise ValueError(f"Unknown label_type: {label_type}")
        
        # Drop rows with NaN labels (at the end of data)
        df = df.dropna(subset=["label"])
        
        return df


# ============================================================================
# Helper Functions
# ============================================================================

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets (chronological).
    
    Args:
        df: Input DataFrame
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        
    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(
        f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df


async def create_historical_klines_table(db_session: AsyncSession) -> None:
    """Create historical_klines table if not exists."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS historical_klines (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(symbol, timeframe, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_klines_symbol_timeframe_timestamp 
        ON historical_klines(symbol, timeframe, timestamp DESC);
    """)
    
    await db_session.execute(create_table_sql)
    await db_session.commit()
    logger.info("Created historical_klines table")

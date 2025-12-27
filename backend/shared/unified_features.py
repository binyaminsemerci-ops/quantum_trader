"""
UNIFIED FEATURE ENGINEERING MODULE
===================================

CRITICAL: This module is used by BOTH training and live inference.
Any changes here affect BOTH systems - be extremely careful!

Features computed:
- 50+ technical indicators
- Price patterns
- Volume indicators  
- Volatility measures
- Momentum indicators

This ensures training and inference use IDENTICAL features.
"""

from typing import Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class UnifiedFeatureConfig:
    """Configuration for feature engineering - shared between training and inference."""
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_periods: list = [9, 21, 50, 200]
    
    # Volume indicators
    volume_sma_period: int = 20
    
    # Volatility
    volatility_window: int = 24
    
    # Price action
    lookback_candles: int = 100


class UnifiedFeatureEngineer:
    """
    CRITICAL: Unified feature engineering for training and inference.
    
    This class MUST produce IDENTICAL features whether called from:
    - backend/domains/learning/ (training)
    - ai_engine/agents/ (live inference)
    
    DO NOT modify without testing BOTH systems!
    """
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        self.config = config or UnifiedFeatureConfig()
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ALL features for model input.
        
        Args:
            df: DataFrame with OHLCV columns (case-insensitive)
            
        Returns:
            DataFrame with 50+ feature columns
        """
        if df is None or df.empty:
            logger.error("CRITICAL: Empty or None DataFrame passed to compute_features")
            return pd.DataFrame()
        
        try:
            # Normalize column names to lowercase
            df = df.copy()
            df.columns = [str(c).lower() for c in df.columns]
            
            # Ensure required columns exist
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    logger.error(f"CRITICAL: Missing required column: {col}")
                    return df
            
            # Sort by timestamp if available
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Compute all feature groups
            df = self._add_price_features(df)
            df = self._add_momentum_indicators(df)
            df = self._add_trend_indicators(df)
            df = self._add_volatility_indicators(df)
            df = self._add_volume_indicators(df)
            df = self._add_pattern_features(df)
            df = self._add_microstructure_features(df)
            
            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.debug(f"Computed {len(df.columns)} features for {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"CRITICAL: Feature computation failed: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-derived features (6 features)."""
        try:
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            df["price_range"] = (df["high"] - df["low"]) / df["close"]
            df["body_size"] = abs(df["close"] - df["open"]) / df["close"]
            df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
            df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
        except Exception as e:
            logger.error(f"Price features failed: {e}")
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, Stochastic, ROC (8 features)."""
        try:
            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
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
            df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
            df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
            
            # Rate of Change
            df["roc"] = (df["close"] - df["close"].shift(10)) / (df["close"].shift(10) + 1e-10) * 100
            
        except Exception as e:
            logger.error(f"Momentum indicators failed: {e}")
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMAs, SMAs, ADX (10+ features)."""
        try:
            # EMAs
            for period in self.config.ema_periods:
                df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
                df[f"ema_{period}_dist"] = (df["close"] - df[f"ema_{period}"]) / df["close"]
            
            # SMA
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            
            # ADX (Average Directional Index)
            high_diff = df["high"].diff()
            low_diff = -df["low"].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = pd.concat([
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-10))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-10))
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df["adx"] = dx.rolling(window=14).mean()
            df["plus_di"] = plus_di
            df["minus_di"] = minus_di
            
        except Exception as e:
            logger.error(f"Trend indicators failed: {e}")
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands, ATR, Volatility (6 features)."""
        try:
            # Bollinger Bands
            bb_middle = df["close"].rolling(self.config.bb_period).mean()
            bb_std = df["close"].rolling(self.config.bb_period).std()
            df["bb_middle"] = bb_middle
            df["bb_upper"] = bb_middle + (bb_std * self.config.bb_std)
            df["bb_lower"] = bb_middle - (bb_std * self.config.bb_std)
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
            
            # ATR
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift(1))
            low_close = abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = tr.rolling(self.config.atr_period).mean()
            df["atr_pct"] = df["atr"] / df["close"]
            
            # Historical volatility
            df["volatility"] = df["returns"].rolling(self.config.volatility_window).std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Volatility indicators failed: {e}")
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features (5 features)."""
        try:
            # Volume SMA and ratio
            df["volume_sma"] = df["volume"].rolling(self.config.volume_sma_period).mean()
            df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)
            
            # OBV (On-Balance Volume)
            obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
            df["obv"] = obv
            df["obv_ema"] = obv.ewm(span=20, adjust=False).mean()
            
            # Volume-Price Trend
            df["vpt"] = ((df["close"].diff() / df["close"].shift(1)) * df["volume"]).cumsum()
            
        except Exception as e:
            logger.error(f"Volume indicators failed: {e}")
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern features (5 features)."""
        try:
            # Doji
            df["is_doji"] = (abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10) < 0.1).astype(int)
            
            # Hammer/Hanging Man
            body = abs(df["close"] - df["open"])
            lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
            df["is_hammer"] = ((lower_shadow > 2 * body) & (body > 0)).astype(int)
            
            # Engulfing
            prev_body = abs(df["close"].shift(1) - df["open"].shift(1))
            curr_body = abs(df["close"] - df["open"])
            df["is_engulfing"] = (curr_body > prev_body * 1.5).astype(int)
            
            # Gap
            df["gap_up"] = (df["low"] > df["high"].shift(1)).astype(int)
            df["gap_down"] = (df["high"] < df["low"].shift(1)).astype(int)
            
        except Exception as e:
            logger.error(f"Pattern features failed: {e}")
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features (5 features)."""
        try:
            # Price momentum
            df["momentum_5"] = df["close"] - df["close"].shift(5)
            df["momentum_10"] = df["close"] - df["close"].shift(10)
            df["momentum_20"] = df["close"] - df["close"].shift(20)
            
            # Acceleration
            df["acceleration"] = df["momentum_5"].diff()
            
            # Relative spread
            df["relative_spread"] = (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2 + 1e-10)
            
        except Exception as e:
            logger.error(f"Microstructure features failed: {e}")
        return df


# Global instance for easy import
_global_engineer = None

def get_feature_engineer() -> UnifiedFeatureEngineer:
    """Get global feature engineer instance."""
    global _global_engineer
    if _global_engineer is None:
        _global_engineer = UnifiedFeatureEngineer()
    return _global_engineer

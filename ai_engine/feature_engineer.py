"""Feature engineering utilities for Quantum Trader.

Enhanced with advanced features for improved prediction accuracy.
Combines basic indicators with 100+ advanced technical features.
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

# Import advanced feature engineering
try:
    from ai_engine.feature_engineer_advanced import add_advanced_features
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False


def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """BULLETPROOF: Compute baseline technical indicators.
    
    NEVER raises, always returns DataFrame (potentially with NaN that are handled).
    Accepts BOTH lowercase ('close') and uppercase ('Close') column names.
    Expected input: DataFrame with OHLCV data
    """
    try:
        df = df.copy()
    except Exception as e:
        print(f"ERROR: Failed to copy dataframe: {e}")
        return pd.DataFrame()  # Return empty DF instead of crashing
    
    try:
        # BULLETPROOF: Detect close column (lowercase or uppercase)
        close_col = None
        if "close" in df.columns:
            close_col = "close"
        elif "Close" in df.columns:
            close_col = "Close"
        else:
            print("ERROR: No 'close' or 'Close' column found")
            return df
        
        # MA indicators with error handling
        try:
            df["ma_10"] = df[close_col].rolling(10, min_periods=1).mean()
        except Exception as e:
            print(f"Warning: ma_10 failed: {e}")
            df["ma_10"] = df[close_col]  # Fallback to close price
        
        try:
            df["ma_50"] = df[close_col].rolling(50, min_periods=1).mean()
        except Exception as e:
            print(f"Warning: ma_50 failed: {e}")
            df["ma_50"] = df[close_col]  # Fallback to close price
        
        try:
            df["rsi_14"] = _rsi(df[close_col], 14)
        except Exception as e:
            print(f"Warning: rsi_14 failed: {e}")
            df["rsi_14"] = 50.0  # Neutral RSI as fallback
        
        return df
    except Exception as e:
        print(f"CRITICAL: compute_basic_indicators crashed: {e}")
        # Return input df unchanged rather than crashing
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def compute_all_indicators(df: pd.DataFrame, use_advanced: bool = True) -> pd.DataFrame:
    """BULLETPROOF: Compute all indicators using UnifiedFeatureEngineer.
    
    NEVER raises, ALWAYS returns valid DataFrame.
    NOW USES UnifiedFeatureEngineer for 50+ CONSISTENT features across training and inference!
    
    Args:
        df: DataFrame with OHLCV data
        use_advanced: Whether to include advanced features (100+)
        
    Returns:
        DataFrame with all indicators (guaranteed non-None)
    """
    # BULLETPROOF: Validate input
    if df is None or not isinstance(df, pd.DataFrame):
        print(f"ERROR: Invalid input to compute_all_indicators: {type(df)}")
        return pd.DataFrame()  # Return empty instead of crashing
    
    if len(df) == 0:
        print("ERROR: Empty DataFrame passed to compute_all_indicators")
        return df
    
    # ✅ USE UNIFIED FEATURE ENGINEER (50+ features, same as training!)
    try:
        from backend.shared.unified_features import get_feature_engineer
        engineer = get_feature_engineer()
        df = engineer.compute_features(df)
        print(f"✅ Computed {len(df.columns)} unified features for {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ UnifiedFeatureEngineer failed: {e}, falling back to legacy")
        # Fallback to old method
    
    # FALLBACK: Compute the 14 EXACT features needed by model
    # This ensures compatibility with trained model
    try:
        df = _compute_model_features(df)
    except Exception as e:
        print(f"CRITICAL: _compute_model_features failed: {e}")
        # Continue with fallback
    
    # Start with basic indicators (guaranteed to work)
    try:
        df = compute_basic_indicators(df)
    except Exception as e:
        print(f"CRITICAL: compute_basic_indicators failed: {e}")
        # Continue with original df
    
    # Add advanced features if available and requested
    if use_advanced and ADVANCED_FEATURES_AVAILABLE:
        try:
            df = add_advanced_features(df)
        except Exception as e:
            print(f"Warning: Could not add advanced features: {e}")
            # Continue without advanced features - not critical
    
    # BULLETPROOF: Ensure we return something
    if df is None or not isinstance(df, pd.DataFrame):
        print("CRITICAL: df became invalid after processing")
        return pd.DataFrame()
    
    return df


def _compute_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the EXACT 14 features the model was trained on.
    
    Required features:
    - Close, Volume, EMA_10, EMA_50, RSI_14
    - MACD, MACD_signal, BB_upper, BB_middle, BB_lower
    - ATR, volume_sma_20, price_change_pct, high_low_range
    """
    try:
        df = df.copy()
        
        # BULLETPROOF: Normalize column names (CSV has lowercase, model expects capitalized)
        for col in ['close', 'high', 'low', 'open', 'volume']:
            if col in df.columns and col.capitalize() not in df.columns:
                df[col.capitalize()] = df[col]
        
        # Ensure we have required columns
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            print("ERROR: Missing Close or Volume columns")
            return df
        
        if 'High' not in df.columns or 'Low' not in df.columns:
            df['High'] = df['Close']
            df['Low'] = df['Close']
        
        # EMA indicators
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        df['RSI_14'] = _rsi(df['Close'], 14)
        
        # MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands (20, 2)
        bb_middle = df['Close'].rolling(20, min_periods=1).mean()
        bb_std = df['Close'].rolling(20, min_periods=1).std()
        df['BB_middle'] = bb_middle
        df['BB_upper'] = bb_middle + (bb_std * 2)
        df['BB_lower'] = bb_middle - (bb_std * 2)
        
        # ATR (14)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
        
        # Volume SMA
        df['volume_sma_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        
        # Price change percentage
        df['price_change_pct'] = df['Close'].pct_change() * 100
        
        # High-Low range
        df['high_low_range'] = df['High'] - df['Low']
        
        # Fill NaN with safe defaults (future-proof against fillna(method) deprecation)
        df = df.ffill().bfill().fillna(0)
        
        return df
    except Exception as e:
        print(f"ERROR in _compute_model_features: {e}")
        return df


def assemble_feature_row(row: pd.Series) -> Dict[str, Any]:
    """Turn a row into a feature dict for model input.

    Keep this function stable (avoid changing keys) so stored artifacts remain
    compatible.
    """
    return {
        "close": row.get("close"),
        "ma_10": row.get("ma_10"),
        "ma_50": row.get("ma_50"),
        "rsi_14": row.get("rsi_14"),
    }


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """BULLETPROOF: Calculate RSI that never fails.
    
    Returns neutral RSI (50) for invalid inputs.
    """
    try:
        # Validate input
        if series is None or len(series) < 2:
            return pd.Series([50.0] * len(series) if len(series) > 0 else [50.0])
        
        # Use numeric coercion to ensure mypy sees a numeric Series
        delta = pd.to_numeric(series.diff(), errors="coerce")
        
        # Handle NaN values
        delta = delta.fillna(0)
        
        up = delta.clip(lower=0)
        down = delta.clip(upper=0).abs()
        
        # Calculate moving averages with min_periods to handle short series
        ma_up = up.rolling(period, min_periods=1).mean()
        ma_down = down.rolling(period, min_periods=1).mean()
        
        # Avoid division by zero
        ma_down = ma_down.replace(0, 0.0001)
        
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        
        # Fill any remaining NaN with neutral value
        rsi = rsi.fillna(50.0)
        
        # Clip to valid range [0, 100]
        rsi = rsi.clip(0, 100)
        
        return rsi
    except Exception as e:
        print(f"ERROR: RSI calculation failed: {e}")
        # Return neutral RSI for all values
        try:
            return pd.Series([50.0] * len(series))
        except:
            return pd.Series([50.0])


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper expected by other modules.

    This currently delegates to `compute_basic_indicators` but exists so the
    training code can call a stable API named `add_technical_indicators`.
    Extend this function with additional indicators as needed.
    """
    return compute_basic_indicators(df)


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_series: pd.Series | None = None,
    news_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """BULLETPROOF: Attach optional sentiment-related features.

    NEVER raises. Conservative: only adds columns when series are provided.
    Handles mismatched lengths and invalid data gracefully.
    """
    try:
        df = df.copy()
    except Exception as e:
        print(f"ERROR: Failed to copy df in add_sentiment_features: {e}")
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    
    try:
        if sentiment_series is not None:
            try:
                # Convert to numeric and handle mismatched lengths
                sentiment_numeric = pd.to_numeric(sentiment_series, errors="coerce")
                
                # Handle length mismatch
                if len(sentiment_numeric) != len(df):
                    # Resize to match df length
                    if len(sentiment_numeric) > len(df):
                        sentiment_numeric = sentiment_numeric[:len(df)]
                    else:
                        # Pad with zeros
                        padding = pd.Series([0.0] * (len(df) - len(sentiment_numeric)))
                        sentiment_numeric = pd.concat([sentiment_numeric, padding], ignore_index=True)
                
                # Fill NaN with neutral sentiment (0.0)
                df["sentiment"] = sentiment_numeric.fillna(0.0)
            except Exception as e:
                print(f"Warning: Failed to add sentiment column: {e}")
                df["sentiment"] = 0.0  # Add neutral sentiment column
        
        if news_counts is not None:
            try:
                # Convert to numeric
                news_numeric = pd.to_numeric(news_counts, errors="coerce")
                
                # Handle length mismatch
                if len(news_numeric) != len(df):
                    if len(news_numeric) > len(df):
                        news_numeric = news_numeric[:len(df)]
                    else:
                        padding = pd.Series([0] * (len(df) - len(news_numeric)))
                        news_numeric = pd.concat([news_numeric, padding], ignore_index=True)
                
                # Fill NaN with 0
                df["news_count"] = news_numeric.fillna(0)
            except Exception as e:
                print(f"Warning: Failed to add news_count column: {e}")
                df["news_count"] = 0  # Add zero news count column
        
        return df
    except Exception as e:
        print(f"CRITICAL: add_sentiment_features crashed: {e}")
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

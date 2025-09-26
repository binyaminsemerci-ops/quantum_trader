"""Feature engineering utilities for Quantum Trader.

This is a small scaffold to be expanded. The goal is to provide a single
place for indicator calculations and feature assembly used by the training
pipeline.
"""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a few baseline technical indicators and return an augmented df.

    Expected input: DataFrame with columns ['open','high','low','close','volume']
    """
    df = df.copy()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = _rsi(df["close"], 14)
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
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # Coerce the price diff to a numeric Series so comparisons are type-safe
    delta = pd.to_numeric(df["Close"].diff(), errors="coerce")
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
    high_low = df["High"] - df["Low"]
    high_close = pd.Series(np.abs(df["High"] - df["Close"].shift()), index=df.index)
    low_close = pd.Series(np.abs(df["Low"] - df["Close"].shift()), index=df.index)
    # Use lambda-based combines (max) to ensure combine receives Series and
    # to avoid numpy.ndarray return types that confuse static type checkers.
    tr1 = high_low.combine(high_close, lambda x, y: x if x >= y else y)
    tr = tr1.combine(low_close, lambda x, y: x if x >= y else y)
    df["ATR"] = tr.rolling(14).mean()
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def add_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.002) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change().shift(-horizon)
    df["Target"] = 0

    # Create a local numeric series to make comparisons type-clear to mypy.
    # Use pd.to_numeric to coerce non-numeric values to NaN (errors='coerce').
    ret = pd.to_numeric(df["Return"], errors="coerce")

    # Now use the typed `ret` series for comparisons. Assign back to df after.
    df.loc[ret > threshold, "Target"] = 1
    df.loc[ret < -threshold, "Target"] = -1

    # Keep Return column as the numeric series
    df["Return"] = ret
    return df.dropna()


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_series: pd.Series | None = None,
    news_counts: pd.Series | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """Add basic sentiment features to the feature DataFrame.

    This is a conservative, small implementation intended to provide a
    stable export for callers (e.g. agents/train code) and to be
    type-checker friendly. It will add two optional columns when
    sentiment or news counts are provided and otherwise return the
    input DataFrame unchanged.
    """
    df = df.copy()
    if sentiment_series is not None:
        # align by index; coerce to numeric safely
        try:
            s = pd.to_numeric(sentiment_series, errors="coerce")
            df[f"sentiment_mean_{window}"] = s.rolling(window=window, min_periods=1).mean().reindex(df.index)
        except Exception:
            # be permissive at runtime but keep typing simple
            df[f"sentiment_mean_{window}"] = sentiment_series.reindex(df.index)

    if news_counts is not None:
        try:
            n = pd.to_numeric(news_counts, errors="coerce")
            df[f"news_count_sum_{window}"] = n.rolling(window=window, min_periods=1).sum().reindex(df.index)
        except Exception:
            df[f"news_count_sum_{window}"] = news_counts.reindex(df.index)

    return df

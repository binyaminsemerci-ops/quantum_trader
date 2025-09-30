"""Feature engineering utilities for Quantum Trader.

The helpers in this module stay intentionally small and composable so they can
be reused by the training CLI, FastAPI routes, and any offline notebooks. When
adding new indicators, place the core transformation here and keep
`train_and_save.py` thin.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_TARGET_COLUMN = "target_return"
DEFAULT_DIRECTION_COLUMN = "target_direction"

__all__ = [
    "compute_basic_indicators",
    "assemble_feature_row",
    "add_technical_indicators",
    "add_sentiment_features",
    "add_target",
]


def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a few baseline technical indicators and return an augmented df.

    Expected input: DataFrame with columns ['open','high','low','close','volume']
    """
    df = df.copy()
    df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()
    df["ma_50"] = df["close"].rolling(50, min_periods=1).mean()
    df["rsi_14"] = _rsi(df["close"], 14)
    return df


def assemble_feature_row(row: pd.Series) -> Dict[str, Any]:
    """Turn a row into a feature dict for model input."""
    return {
        "close": row.get("close"),
        "ma_10": row.get("ma_10"),
        "ma_50": row.get("ma_50"),
        "rsi_14": row.get("rsi_14"),
        "sentiment": row.get("sentiment"),
        "news_count": row.get("news_count"),
    }


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = pd.to_numeric(series.diff(), errors="coerce")
    up = delta.clip(lower=0)
    down = delta.clip(upper=0).abs()
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper expected by other modules."""
    return compute_basic_indicators(df)


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_series: Optional[Iterable[float]] = None,
    news_counts: Optional[Iterable[float]] = None,
    *,
    window: int = 1,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Attach optional sentiment-related features to the dataframe."""
    df = df.copy()
    if sentiment_series is not None:
        sentiment = sentiment_series if isinstance(sentiment_series, pd.Series) else pd.Series(list(sentiment_series))
        sentiment = pd.to_numeric(sentiment.reindex(df.index, fill_value=fill_value), errors="coerce")
        if window > 1:
            sentiment = sentiment.rolling(window, min_periods=1).mean()
        df["sentiment"] = sentiment
    if news_counts is not None:
        news = news_counts if isinstance(news_counts, pd.Series) else pd.Series(list(news_counts))
        news = pd.to_numeric(news.reindex(df.index, fill_value=0), errors="coerce")
        if window > 1:
            news = news.rolling(window, min_periods=1).sum()
        df["news_count"] = news
    return df


def add_target(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    target_column: str = DEFAULT_TARGET_COLUMN,
    direction_column: str = DEFAULT_DIRECTION_COLUMN,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Add forward-looking return and direction targets.

    Parameters
    ----------
    df:
        DataFrame that must contain a `close` column.
    horizon:
        Number of rows to look forward when computing returns.
    target_column:
        Column name to store the fractional return.
    direction_column:
        Column name for the long/short/flat signal derived from the return.
    threshold:
        Minimum absolute return required before signalling a position. Values
        inside ``(-threshold, threshold)`` produce a flat (0) signal.
    """

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column to compute targets")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    df_out = df.copy()
    close = pd.to_numeric(df_out["close"], errors="coerce")
    future_close = close.shift(-horizon)
    returns = (future_close - close) / close.replace(0, np.nan)
    df_out[target_column] = returns

    if direction_column:
        df_out[direction_column] = np.where(
            returns > threshold,
            1,
            np.where(returns < -threshold, -1, 0),
        )

    if horizon > 0:
        df_out = df_out.iloc[:-horizon]
    df_out = df_out.dropna(subset=[target_column])
    return df_out

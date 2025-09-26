"""Feature engineering utilities for Quantum Trader.

This is a small scaffold to be expanded. The goal is to provide a single
place for indicator calculations and feature assembly used by the training
pipeline.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

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
    # Use numeric coercion to ensure mypy sees a numeric Series
    delta = pd.to_numeric(series.diff(), errors="coerce")
    up = delta.clip(lower=0)
    # Use abs on clipped negative moves to get positive losses; this keeps types numeric
    down = delta.clip(upper=0).abs()
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible alias expected by other modules.

    Delegates to `compute_basic_indicators` for now; keep this stable so
    callers (including agents and training code) can rely on the name.
    """
    return compute_basic_indicators(df)


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    news_counts: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Attach optional sentiment-related features to the dataframe.

    This is intentionally conservative: it only adds columns when
    corresponding series are provided and returns the augmented DataFrame.
    """
    df = df.copy()
    if sentiment_series is not None:
        df["sentiment"] = pd.to_numeric(sentiment_series, errors="coerce")
    if news_counts is not None:
        df["news_count"] = pd.to_numeric(news_counts, errors="coerce")
    return df

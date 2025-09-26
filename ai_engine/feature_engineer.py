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
    # Use numeric coercion to ensure mypy sees a numeric Series
    delta = pd.to_numeric(series.diff(), errors="coerce")
    up = delta.clip(lower=0)
    # Use abs on clipped negative moves to get positive losses; this keeps types numeric
    down = delta.clip(upper=0).abs()
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
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
    # Ensure Return series is numeric so static analysis understands the
    # comparisons against `threshold` are valid (pandas stubs can report
    # Series[object] which doesn't support >/< with numeric literals).
    try:
        df["Return"] = pd.to_numeric(df["Return"], errors="coerce")
    except Exception:
        # If conversion fails at runtime, fall back to original object series
        pass

    df.loc[df["Return"] > threshold, "Target"] = 1
    df.loc[df["Return"] < -threshold, "Target"] = -1
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

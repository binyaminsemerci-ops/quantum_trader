import pandas as pd  # type: ignore[import-untyped]
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    # cast intermediate Series to pd.Series to satisfy strict pandas stubs
    from typing import cast
    delta_series = cast(pd.Series, delta)
    # ensure comparisons are done against floats so stubs accept the operation
    gain = cast(pd.Series, (delta_series.where(delta_series > 0.0, 0.0))).rolling(14).mean()
    loss = cast(pd.Series, (-delta_series.where(delta_series < 0.0, 0.0))).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
    high_low = cast(pd.Series, df["High"] - df["Low"])
    high_close = cast(pd.Series, np.abs(df["High"] - df["Close"].shift()))
    low_close = cast(pd.Series, np.abs(df["Low"] - df["Close"].shift()))
    # combine expects Series inputs; ensure we pass Series by casting numpy arrays back when needed
    # Use explicit Series constructors when numpy operations may produce ndarray
    high_close_s = cast(pd.Series, pd.Series(high_close))
    low_close_s = cast(pd.Series, pd.Series(low_close))
    tr = cast(pd.Series, high_low.combine(high_close_s, np.maximum).combine(low_close_s, np.maximum))
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
    df.loc[df["Return"] > threshold, "Target"] = 1
    df.loc[df["Return"] < -threshold, "Target"] = -1
    return df.dropna()


def add_sentiment_features(df: pd.DataFrame, sentiment_series=None, news_counts=None, window: int = 5) -> pd.DataFrame:
    """Append simple sentiment-based features to a price dataframe.

    - sentiment_series: pd.Series aligned by timestamp (same length) with float scores
    - news_counts: pd.Series or list-like aligned with counts of news items per row
    - window: rolling window (in rows) to aggregate sentiment/news counts
    """
    df = df.copy()
    n = len(df)

    # align sentiment
    if sentiment_series is None:
        df["sentiment_mean"] = 0.0
        df["sentiment_std"] = 0.0
    else:
        s = None
        try:
            import pandas as _pd

            if isinstance(sentiment_series, list):
                s = _pd.Series(sentiment_series)
            elif hasattr(sentiment_series, 'values'):
                s = _pd.Series(sentiment_series.values)
            else:
                s = _pd.Series(sentiment_series)
        except Exception:
            # fallback: try to coerce to list
            s = pd.Series(list(sentiment_series))

        # pad/trim to match df length
        if len(s) < n:
            s = _pd.concat([_pd.Series([0.0] * (n - len(s))), s], ignore_index=True)
        elif len(s) > n:
            s = s.iloc[-n:].reset_index(drop=True)

        df["sentiment_mean"] = s.rolling(window=window, min_periods=1).mean().values
        df["sentiment_std"] = s.rolling(window=window, min_periods=1).std().fillna(0).values

    # news count
    if news_counts is None:
        df["news_count"] = 0
        df["news_count_roll"] = 0
    else:
        try:
            import pandas as _pd

            nc = _pd.Series(news_counts)
        except Exception:
            nc = pd.Series(list(news_counts))

        if len(nc) < n:
            nc = _pd.concat([_pd.Series([0] * (n - len(nc))), nc], ignore_index=True)
        elif len(nc) > n:
            nc = nc.iloc[-n:].reset_index(drop=True)

        df["news_count"] = nc.values
        df["news_count_roll"] = nc.rolling(window=window, min_periods=1).sum().values

    return df

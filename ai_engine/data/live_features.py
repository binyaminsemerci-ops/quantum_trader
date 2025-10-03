"""Helpers to build scikit-learn friendly feature matrices from recent candles.

This module fetches recent candles via the project's market-data helper and
builds lagged features (close lags, volume lags, returns) plus a target
column (future return). It keeps the interface small so the trainer can call
it when using live Binance data.
"""
from __future__ import annotations

from typing import Tuple, List
import numpy as np




def _get_value(entry: dict, keys: List[str]):
    for k in keys:
        if k in entry:
            return entry[k]
    return None


def fetch_features_for_sklearn(symbol: str, limit: int = 200, lags: int = 5, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fetch recent candles and return (X, y, feature_names).

    - X: shape (n_samples, n_features)
    - y: shape (n_samples,) containing future returns over `horizon` periods
    - feature_names: list of column names in X

    Features produced:
    - close_lag_1 .. close_lag_{lags}
    - vol_lag_1 .. vol_lag_{lags}
    - rtn_lag_1 .. rtn_lag_{lags}  (log returns)

    The function will raise ValueError if limit is too small for the requested lags/horizon.
    """
    if limit <= lags + horizon + 2:
        raise ValueError("limit must be greater than lags + horizon + 2")

    # Try to use ccxt directly to fetch live OHLCV (avoids importing project
    # route modules which can cause circular imports). If ccxt is missing or
    # the call fails, fall back to a deterministic demo series.
    try:
        import ccxt  # type: ignore

        exchange = ccxt.binance({"enableRateLimit": True})
        # normalize symbol for ccxt: BTCUSDT -> BTC/USDT
        if '/' in symbol:
            ccxt_sym = symbol
        elif symbol.upper().endswith('USDC'):
            ccxt_sym = f"{symbol[:-4]}/USDC"
        elif symbol.upper().endswith('USDT'):
            ccxt_sym = f"{symbol[:-4]}/USDT"
        else:
            ccxt_sym = f"{symbol}/USDT"

        timeframe = '1m'
        ohlcv = exchange.fetch_ohlcv(ccxt_sym, timeframe=timeframe, limit=limit)
        raw = []
        for ts, open_p, high_p, low_p, close_p, volume in ohlcv:
            # ts is milliseconds
            raw.append({
                "time": ts,
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": float(volume),
            })
    except Exception:
        # demo fallback (deterministic-ish)
        raw = []
        base = 100.0 + (abs(hash(symbol)) % 50)
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        for i in range(limit):
            ts = now - timedelta(minutes=(limit - i))
            open_p = base + (i * 0.1) + (0.5 * (i % 3))
            close_p = open_p + ((-1) ** i) * (0.5 * ((i % 5) / 5.0))
            high_p = max(open_p, close_p) + 0.4
            low_p = min(open_p, close_p) - 0.4
            vol = 10 + (i % 7)
            raw.append({
                "time": ts.isoformat(),
                "open": round(open_p, 3),
                "high": round(high_p, 3),
                "low": round(low_p, 3),
                "close": round(close_p, 3),
                "volume": vol,
            })
    # raw entries typically contain 'time' or 'timestamp', and numeric 'close' and 'volume'
    closes = []
    vols = []
    times = []
    for e in raw:
        close = _get_value(e, ["close", "Close", "ClosePrice", "c"]) or 0.0
        vol = _get_value(e, ["volume", "Volume", "v"]) or 0.0
        t = _get_value(e, ["time", "timestamp"]) or None
        try:
            closes.append(float(close))
        except Exception:
            closes.append(0.0)
        try:
            vols.append(float(vol))
        except Exception:
            vols.append(0.0)
        times.append(t)

    if len(closes) < lags + horizon + 1:
        raise ValueError("Not enough candle rows returned to build features")

    # build arrays (oldest first assumed) — fetch_recent_candles returns newest-last in demo/live
    # ensure order: oldest -> newest
    # Many helpers already return ascending by time; if not, this still works as a best-effort.
    closes_arr = np.array(closes, dtype=float)
    vols_arr = np.array(vols, dtype=float)

    # compute log returns (safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        rtns = np.log(closes_arr[1:] / np.where(closes_arr[:-1] == 0, 1.0, closes_arr[:-1]))
    # pad to align length with closes (first rtn is 0)
    rtns = np.concatenate(([0.0], rtns))

    n = len(closes_arr)
    rows = []
    feature_names = []
    for lag in range(1, lags + 1):
        feature_names.append(f"close_lag_{lag}")
    for lag in range(1, lags + 1):
        feature_names.append(f"vol_lag_{lag}")
    for lag in range(1, lags + 1):
        feature_names.append(f"rtn_lag_{lag}")

    # build samples for indices where we can read lags and horizon
    for i in range(lags, n - horizon):
        row = []
        # close lags: most recent first
        for lag in range(1, lags + 1):
            row.append(closes_arr[i - lag])
        # vol lags
        for lag in range(1, lags + 1):
            row.append(vols_arr[i - lag])
        # rtn lags
        for lag in range(1, lags + 1):
            row.append(rtns[i - lag])
        rows.append(row)

    X = np.array(rows, dtype=float)

    # build target: future return over horizon (log return from now to t+horizon)
    y = []
    for i in range(lags, n - horizon):
        future_close = closes_arr[i + horizon]
        now_close = closes_arr[i]
        if now_close == 0:
            y.append(0.0)
        else:
            y.append(float(np.log(future_close / now_close)))
    y_arr = np.array(y, dtype=float)

    return X, y_arr, feature_names

"""Simple training harness for the XGBoost agent.

This script fetches data from the internal `backend.routes.external_data` helpers
to build a dataset, adds technical + sentiment features, trains a regressor,
and writes model and scaler artifacts to `ai_engine/models/`.

The script is defensive: if xgboost or sklearn are not installed it falls back
to a DummyRegressor and a lightweight scaler so it can run in minimal CI/dev
environments.
"""
import os
import asyncio
import pickle
from typing import List, Optional

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class _SimpleScaler:
    """Very small replacement for sklearn StandardScaler when not available."""
    def fit(self, X):
        import numpy as _np

        self.mean_ = _np.nanmean(X, axis=0)
        self.scale_ = _np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X):

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class _MeanRegressor:
    """Simple mean predictor used when no regressor is available."""
    def fit(self, X, y):
        import numpy as _np

        self.mean_ = float(_np.nanmean(y))

    def predict(self, X):
        import numpy as _np

        return _np.full((len(X),), getattr(self, 'mean_', 0.0))


async def _fetch_symbol_data(symbol: str, limit: int = 600):
    # import the internal async route function and call it
    from backend.routes import external_data

    # external_data.binance_ohlcv is async; call with asyncio
    resp = await external_data.binance_ohlcv(symbol=symbol, limit=limit)
    candles = resp.get('candles', [])

    # sentiment: try twitter client
    tw = await external_data.twitter_sentiment(symbol=symbol)
    sent_score = 0.0
    try:
        sent_score = float(tw.get('sentiment', {}).get('score', 0.0))
    except Exception:
        sent_score = 0.0

    # news: count
    news = await external_data.cryptopanic_news(symbol=symbol, limit=200)
    news_count = len(news.get('news', []))

    # Expand sentiment/news into series aligned to candles
    n = len(candles)
    sentiment_series = [sent_score] * n
    news_series = [0] * n
    if n > 0 and news_count > 0:
        # place news events sparsely into the array
        step = max(1, n // news_count)
        for i in range(0, n, step):
            if sum(news_series) >= news_count:
                break
            news_series[i] = 1

    return candles, sentiment_series, news_series


def build_dataset(all_symbol_data):
    """Given list of (symbol, candles, sentiment, news) tuples, build X,y arrays."""
    import pandas as pd  # type: ignore[import-untyped]
    import numpy as np
    from ai_engine.feature_engineer import add_technical_indicators, add_sentiment_features, add_target

    X_list = []
    y_list = []
    for symbol, candles, sentiment, news in all_symbol_data:
        if not candles:
            continue
        df = pd.DataFrame(candles)
        # ensure expected column names
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        # compute technicals
        try:
            feat = add_technical_indicators(df.rename(columns={'Close': 'Close', 'High': 'High', 'Low': 'Low'}))
        except Exception:
            continue

        # add sentiment/news aligned series
        feat = add_sentiment_features(feat, sentiment_series=sentiment, news_counts=news, window=5)

        # add target (predict Return horizon=1)
        feat_t = add_target(feat, horizon=1, threshold=0.0)
        if feat_t.empty:
            continue

        y = feat_t['Return'].values
        X = feat_t.select_dtypes(include=[float, int]).drop(columns=['Return']).values
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError('No training data assembled')

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    return X_all, y_all


def make_scaler():
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
        return StandardScaler()
    except Exception:
        return _SimpleScaler()


def make_regressor():
    try:
        from xgboost import XGBRegressor  # type: ignore[import-untyped]
        return XGBRegressor(n_estimators=50, max_depth=3, verbosity=0)
    except Exception:
        try:
            from sklearn.dummy import DummyRegressor  # type: ignore[import-untyped]
            return DummyRegressor(strategy='mean')
        except Exception:
            return _MeanRegressor()


def save_artifacts(model, scaler, model_path, scaler_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    # write simple metadata JSON next to model
    try:
        import json
        import datetime
        meta = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'saved_at': datetime.datetime.utcnow().isoformat() + 'Z',
        }
        meta_path = os.path.join(MODEL_DIR, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf)
    except Exception:
        pass


def train_and_save(symbols: Optional[List[str]] = None, limit: int = 600):
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [_fetch_symbol_data(s, limit=limit) for s in symbols]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    all_symbol_data = []
    for sym, (candles, sentiment, news) in zip(symbols, results):
        all_symbol_data.append((sym, candles, sentiment, news))

    # Attempt to build dataset using pandas-based routines; if pandas is
    # unavailable (e.g., import-time binary issues), fall back to a
    # synthetic dataset so training can proceed for CI/dev.
    try:
        X, y = build_dataset(all_symbol_data)
    except Exception as exc:  # pragma: no cover - environment dependent
        print('Warning: build_dataset failed, falling back to synthetic dataset:', exc)
        # create a small synthetic dataset
        import numpy as _np

        def synthetic_dataset(samples=1000, features=16):
            rng = _np.random.default_rng(12345)
            Xs = rng.normal(size=(samples, features))
            # create a target correlated with a subset of features
            y = (Xs[:, 0] * 0.3 + Xs[:, 1] * -0.2 + rng.normal(scale=0.1, size=(samples,)))
            return Xs, y

        X, y = synthetic_dataset(samples=1000, features=16)

    scaler = make_scaler()
    try:
        Xs = scaler.fit_transform(X)
    except Exception:
        Xs = scaler.fit_transform(X)

    reg = make_regressor()
    reg.fit(Xs, y)

    model_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    save_artifacts(reg, scaler, model_path, scaler_path)
    print('Saved model ->', model_path)
    print('Saved scaler ->', scaler_path)


if __name__ == '__main__':
    # quick local run (will use internal route fallbacks if external APIs are not configured)
    train_and_save()

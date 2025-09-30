from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config.config import DEFAULT_SYMBOLS, settings
from ai_engine.feature_engineer import (
    DEFAULT_TARGET_COLUMN,
    add_sentiment_features,
    add_target,
    add_technical_indicators,
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILENAME = "training_report.json"
logger = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    """Container holding feature arrays and metadata for training/backtests."""

    features: np.ndarray
    target: np.ndarray
    timestamps: List[str]
    feature_names: List[str]


class _SimpleScaler:
    """Very small replacement for sklearn StandardScaler when not available."""

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class _MeanRegressor:
    """Simple mean predictor used when no regressor is available."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = float(np.nanmean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X),), getattr(self, "mean_", 0.0), dtype=float)


async def _fetch_symbol_data(symbol: str, limit: int = 600) -> Tuple[List[dict], List[float], List[float]]:
    """Fetch candles + sentiment/news summaries using internal stubs."""
    from backend.routes import external_data

    candles_payload = await external_data.binance_ohlcv(symbol=symbol, limit=limit)
    candles = candles_payload.get("candles", [])

    sentiment_payload = await external_data.twitter_sentiment(symbol=symbol)
    sent_score = 0.0
    try:
        sent_score = float(sentiment_payload.get("sentiment", {}).get("score", sentiment_payload.get("score", 0.0)))
    except Exception:
        sent_score = 0.0

    news_payload = await external_data.cryptopanic_news(symbol=symbol, limit=200)
    news_count = len(news_payload.get("news", []))

    n = len(candles)
    sentiment_series = [sent_score] * n
    news_series = [0.0] * n
    if n > 0 and news_count > 0:
        step = max(1, n // news_count)
        inserted = 0
        for idx in range(0, n, step):
            if inserted >= news_count:
                break
            news_series[idx] = 1.0
            inserted += 1

    return candles, sentiment_series, news_series


def _gather_symbol_payloads(symbols: Sequence[str], limit: int) -> List[Tuple[str, List[dict], List[float], List[float]]]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [_fetch_symbol_data(symbol, limit=limit) for symbol in symbols]
    try:
        results = loop.run_until_complete(asyncio.gather(*tasks))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
    grouped: List[Tuple[str, List[dict], List[float], List[float]]] = []
    for symbol, payload in zip(symbols, results):
        grouped.append((symbol, payload[0], payload[1], payload[2]))
    return grouped


def build_dataset(
    all_symbol_data: Iterable[Tuple[str, List[dict], List[float], List[float]]],
    *,
    horizon: int = 1,
) -> PreparedDataset:
    """Transform fetched symbol data into feature/target arrays."""
    import pandas as pd

    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    timestamps: List[str] = []
    feature_names: List[str] = []

    for _, candles, sentiment, news in all_symbol_data:
        if not candles:
            continue
        df = pd.DataFrame(candles).rename(columns=lambda c: str(c).lower())
        if "close" not in df.columns:
            continue
        df = add_technical_indicators(df)
        df = add_sentiment_features(df, sentiment_series=sentiment, news_counts=news, window=5)
        df = add_target(df, horizon=horizon, target_column=DEFAULT_TARGET_COLUMN, threshold=0.0)
        if df.empty or DEFAULT_TARGET_COLUMN not in df.columns:
            continue

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if DEFAULT_TARGET_COLUMN not in numeric_df.columns:
            continue
        y = numeric_df[DEFAULT_TARGET_COLUMN].to_numpy(dtype=float)
        feature_df = numeric_df.drop(columns=[DEFAULT_TARGET_COLUMN])
        if feature_df.empty:
            continue
        feature_names = list(feature_df.columns)
        X = feature_df.to_numpy(dtype=float)
        if not len(X):
            continue

        X_blocks.append(X)
        y_blocks.append(y)

        ts_series = df["timestamp"] if "timestamp" in df.columns else pd.Series(df.index.astype(str))
        ts_values = [str(v) for v in ts_series.iloc[: len(y)]]
        timestamps.extend(ts_values)

    if not X_blocks:
        raise RuntimeError("No training data assembled from the provided symbols")

    features = np.vstack(X_blocks)
    target = np.concatenate(y_blocks)
    if not timestamps:
        timestamps = [f"sample-{i}" for i in range(len(target))]
    return PreparedDataset(features=features, target=target, timestamps=timestamps, feature_names=feature_names)


def _synthetic_dataset(samples: int = 1000, features: int = 16) -> PreparedDataset:
    rng = np.random.default_rng(12345)
    X = rng.normal(size=(samples, features))
    y = X[:, 0] * 0.3 + X[:, 1] * -0.2 + rng.normal(scale=0.1, size=(samples,))
    timestamps = [f"synthetic-{i}" for i in range(samples)]
    feature_names = [f"f{i}" for i in range(features)]
    return PreparedDataset(features=X, target=y.astype(float), timestamps=timestamps, feature_names=feature_names)


def make_scaler():
    try:
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    except Exception:
        return _SimpleScaler()


def make_regressor():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=50, max_depth=3, verbosity=0)
    except Exception:
        try:
            from sklearn.dummy import DummyRegressor

            return DummyRegressor(strategy="mean")
        except Exception:
            return _MeanRegressor()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    direction = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": direction,
    }


def simulate_equity_curve(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    timestamps: Sequence[str],
    *,
    starting_equity: float = 10000.0,
    entry_threshold: float = 0.001,
    fee_per_trade: float = 0.0005,
    max_abs_return: float = 0.01,
) -> dict:
    equity = starting_equity
    curve: List[dict] = []
    wins = 0
    losses = 0
    trades = 0
    peak = starting_equity
    max_drawdown = 0.0

    for ts, true_ret, pred_ret in zip(timestamps, y_true, y_pred):
        signal = 1 if pred_ret > entry_threshold else -1 if pred_ret < -entry_threshold else 0
        trade_return = 0.0
        if signal != 0:
            trades += 1
            raw_return = signal * float(true_ret)
            clamped_return = float(np.clip(raw_return, -max_abs_return, max_abs_return))
            trade_return = clamped_return - fee_per_trade
            if trade_return > 0:
                wins += 1
            elif trade_return < 0:
                losses += 1
            equity *= 1 + trade_return
        curve.append({"timestamp": str(ts), "equity": round(equity, 2), "signal": signal})
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    pnl = equity - starting_equity
    win_rate = wins / trades if trades else 0.0
    return {
        "starting_equity": starting_equity,
        "final_equity": round(equity, 2),
        "pnl": round(pnl, 2),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4),
        "equity_curve": curve,
    }


def save_artifacts(model, scaler, model_path: Path, scaler_path: Path) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts: dict = {}
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        artifacts["model_path"] = str(model_path)
    except Exception as exc:
        fallback = model_path.with_suffix(".json")
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump({"error": str(exc)}, f)
        artifacts["model_path"] = str(fallback)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    artifacts["scaler_path"] = str(scaler_path)

    metadata_path = model_path.parent / "metadata.json"
    metadata = {
        "model_path": artifacts["model_path"],
        "scaler_path": artifacts["scaler_path"],
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf)
    artifacts["metadata_path"] = str(metadata_path)
    return artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _prepare_dataset(symbols: Sequence[str], limit: int) -> PreparedDataset:
    try:
        symbol_payloads = _gather_symbol_payloads(symbols, limit)
        return build_dataset(symbol_payloads)
    except Exception as exc:
        logger.warning("Falling back to synthetic dataset: %s", exc)
        return _synthetic_dataset()


def train_and_save(
    symbols: Optional[Sequence[str]] = None,
    limit: int = 600,
    *,
    model_dir: Optional[Path | str] = None,
    backtest: bool = True,
    write_report: bool = True,
    entry_threshold: float = 0.001,
) -> dict:
    if symbols is None:
        default_symbols = list(DEFAULT_SYMBOLS)
        if default_symbols:
            symbols = default_symbols
        else:
            symbols = [f"BTC{settings.default_quote}", f"ETH{settings.default_quote}"]

    dataset = _prepare_dataset(list(symbols), limit)

    scaler = make_scaler()
    features_scaled = scaler.fit_transform(dataset.features)

    regressor = make_regressor()
    regressor.fit(features_scaled, dataset.target)

    predictions = np.asarray(regressor.predict(features_scaled), dtype=float)
    metrics = evaluate_predictions(dataset.target, predictions)

    backtest_report = None
    if backtest:
        backtest_report = simulate_equity_curve(
            dataset.target,
            predictions,
            dataset.timestamps,
            entry_threshold=entry_threshold,
        )

    output_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_path = output_dir / "xgb_model.pkl"
    scaler_path = output_dir / "scaler.pkl"
    artifacts = save_artifacts(regressor, scaler, model_path, scaler_path)

    report_payload = {
        "symbols": list(symbols),
        "limit": limit,
        "num_samples": int(dataset.target.shape[0]),
        "feature_names": dataset.feature_names,
        "metrics": metrics,
        "entry_threshold": entry_threshold,
    }
    if backtest_report is not None:
        report_payload["backtest"] = backtest_report

    report_path = output_dir / REPORT_FILENAME if write_report else None
    if report_path is not None:
        _write_json(report_path, report_payload)

    return {
        "artifacts": artifacts,
        "metrics": metrics,
        "backtest": backtest_report,
        "report_path": str(report_path) if report_path is not None else None,
    }


def load_artifacts(model_dir: Optional[Path | str] = None) -> Tuple[object, object]:
    directory = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_path = directory / "xgb_model.pkl"
    scaler_path = directory / "scaler.pkl"
    if not model_path.exists() and model_path.with_suffix(".json").exists():
        model_path = model_path.with_suffix(".json")
    with open(model_path, "rb" if model_path.suffix == ".pkl" else "r") as f:
        if model_path.suffix == ".pkl":
            model = pickle.load(f)
        else:
            spec = json.load(f)

            class JSONModel:
                def __init__(self, spec: dict) -> None:
                    self.scale = float(spec.get("scale", 1.0))

                def predict(self, arr):
                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return np.nanmean(arr, axis=1) * self.scale

            model = JSONModel(spec)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def run_backtest_only(
    symbols: Sequence[str],
    limit: int = 600,
    *,
    model_dir: Optional[Path | str] = None,
    entry_threshold: float = 0.001,
) -> dict:
    dataset = _prepare_dataset(list(symbols), limit)
    model, scaler = load_artifacts(model_dir)
    features_scaled = scaler.transform(dataset.features)
    predictions = np.asarray(model.predict(features_scaled), dtype=float)
    metrics = evaluate_predictions(dataset.target, predictions)
    backtest_report = simulate_equity_curve(
        dataset.target,
        predictions,
        dataset.timestamps,
        entry_threshold=entry_threshold,
    )
    return {
        "metrics": metrics,
        "backtest": backtest_report,
        "num_samples": int(dataset.target.shape[0]),
    }


def load_report(model_dir: Optional[Path | str] = None) -> Optional[dict]:
    directory = Path(model_dir) if model_dir is not None else MODEL_DIR
    path = directory / REPORT_FILENAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    summary = train_and_save()
    print(json.dumps(summary, indent=2))

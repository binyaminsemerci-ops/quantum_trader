"""Data ingestion and feature engineering pipeline for model training.

This module can be run as a script or imported for orchestration in other
workflows. Logging output summarises progress while Prometheus counters provide
basic observability for scheduled runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Tuple

from prometheus_client import Counter, Histogram, CollectorRegistry

from backend.utils.telemetry import track_model_inference  # reuse timing helper

# Use custom registry to avoid conflicts
_pipeline_registry = CollectorRegistry()

RAW_ROWS_COLLECTED = Counter(
    "qt_pipeline_raw_rows_total",
    "Number of raw rows collected per symbol",
    labelnames=("symbol",),
    registry=_pipeline_registry,
)
RAW_INGEST_DURATION = Histogram(
    "qt_pipeline_raw_ingest_duration_seconds",
    "Duration of raw market data fetches",
    labelnames=("symbol",),
    registry=_pipeline_registry,
)
FEATURE_ROWS_WRITTEN = Counter(
    "qt_pipeline_feature_rows_total",
    "Number of feature rows written per dataset",
    labelnames=("dataset",),
    registry=_pipeline_registry,
)


async def ingest_market_data(
    symbols: Iterable[str],
    *,
    limit: int = 600,
    output_dir: Path,
) -> Path:
    """Fetch market + sentiment payloads for the given symbols."""

    output_dir.mkdir(parents=True, exist_ok=True)

    async def _fetch(symbol: str) -> Tuple[str, dict]:
        from backend.routes.external_data import binance_ohlcv, twitter_sentiment

        symbol = symbol.upper()
        start = perf_counter()
        candles_payload = await binance_ohlcv(symbol=symbol, limit=limit)
        duration = perf_counter() - start
        RAW_INGEST_DURATION.labels(symbol=symbol).observe(duration)

        sentiment = await twitter_sentiment(symbol)
        candles = candles_payload.get("candles", [])
        RAW_ROWS_COLLECTED.labels(symbol=symbol).inc(len(candles))
        logging.info("Fetched %s rows for %s in %.2fs", len(candles), symbol, duration)
        return symbol, {"candles": candles, "sentiment": sentiment}

    results = await asyncio.gather(*[_fetch(sym) for sym in symbols])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = output_dir / f"market_raw_{timestamp}.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump({symbol: payload for symbol, payload in results}, handle, indent=2)
    logging.info("Wrote raw dataset -> %s", raw_path)
    return raw_path


def build_feature_matrix(
    raw_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Transform raw ingestion payload into a feature matrix."""

    import pandas as pd  # type: ignore[import-untyped]
    from ai_engine.feature_engineer import add_technical_indicators, add_sentiment_features

    with raw_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frames: List[pd.DataFrame] = []
    for symbol, content in payload.items():
        candles = content.get("candles", [])
        if not candles:
            continue
        df = pd.DataFrame(candles)
        if df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        feat = add_technical_indicators(df)
        sentiment = content.get("sentiment", {})
        sentiment_value = sentiment.get("score") or sentiment.get("sentiment", {}).get("score")
        if sentiment_value is not None:
            feat = add_sentiment_features(feat, sentiment_series=pd.Series(sentiment_value, index=feat.index))
        feat["symbol"] = symbol
        frames.append(feat)

    dataset = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if dataset.empty:
        logging.warning("No feature rows generated from %s", raw_path)
    dataset["as_of"] = datetime.now(timezone.utc)

    output_path = output_path or raw_path.with_suffix(".features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    FEATURE_ROWS_WRITTEN.labels(dataset=output_path.name).inc(len(dataset))
    logging.info("Wrote feature matrix (%s rows) -> %s", len(dataset), output_path)
    return output_path


async def run_pipeline(args: argparse.Namespace) -> None:
    """Execute ingestion + feature generation."""

    raw_path = await ingest_market_data(args.symbols, limit=args.limit, output_dir=args.output_dir)
    # reuse telemetry helper purely for timing measurements
    with track_model_inference("feature_pipeline"):
        build_feature_matrix(raw_path, output_path=args.features_path)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest market data and build feature matrix")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to ingest")
    parser.add_argument("--limit", type=int, default=600, help="Candle limit per symbol")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "datasets",
        help="Directory to write raw dataset files",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Optional explicit path for feature matrix",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()

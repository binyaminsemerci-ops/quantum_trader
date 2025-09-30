"""Async helpers providing market/sentiment data for training/backtests.

The original demo returned hard-coded payloads. This version routes requests
through the shared config + adapter layer when possible and keeps deterministic
fallbacks so CI/dev environments without external credentials still behave.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from config.config import load_config, settings
from backend.routes.settings import SETTINGS
from backend.utils.market_data import fetch_recent_candles
from backend.utils.twitter_client import TwitterClient
from backend.utils.cryptopanic_client import CryptoPanicClient

logger = logging.getLogger(__name__)

_TWITTER_CLIENT: TwitterClient | None = None
_CRYPTOPANIC_CLIENT: CryptoPanicClient | None = None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _live_market_data_enabled() -> bool:
    override = SETTINGS.get("ENABLE_LIVE_MARKET_DATA")
    if override is not None:
        return _coerce_bool(override)
    cfg = load_config()
    return bool(cfg.enable_live_market_data)


def _strip_quote(symbol: str) -> str:
    quote = load_config().default_quote or settings.default_quote
    upper = symbol.upper()
    if upper.endswith(quote.upper()):
        return symbol[: -len(quote)]
    return symbol


def _fallback_candles(symbol: str, limit: int) -> List[Dict[str, Any]]:
    candles: List[Dict[str, Any]] = []
    price = 100.0
    for idx in range(limit):
        close_val = price + ((idx % 3) - 1) * 0.1
        candles.append(
            {
                "timestamp": f"t{idx}",
                "time": f"t{idx}",
                "open": float(price),
                "high": float(price + 1),
                "low": float(price - 1),
                "close": float(close_val),
                "volume": float(100 + idx),
            }
        )
        price = close_val
    return candles


def _normalise_candles(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for entry in raw:
        timestamp = entry.get("timestamp") or entry.get("time")
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        def _as_float(key: str) -> float:
            try:
                return float(entry.get(key, 0.0))
            except Exception:
                return 0.0

        normalised.append(
            {
                "timestamp": str(timestamp),
                "open": _as_float("open"),
                "high": _as_float("high"),
                "low": _as_float("low"),
                "close": _as_float("close"),
                "volume": _as_float("volume"),
            }
        )
    return normalised


def _twitter_client() -> TwitterClient:
    global _TWITTER_CLIENT
    if _TWITTER_CLIENT is None:
        _TWITTER_CLIENT = TwitterClient()
    return _TWITTER_CLIENT


def _cryptopanic_client() -> CryptoPanicClient:
    global _CRYPTOPANIC_CLIENT
    if _CRYPTOPANIC_CLIENT is None:
        _CRYPTOPANIC_CLIENT = CryptoPanicClient()
    return _CRYPTOPANIC_CLIENT


def _fallback_news(symbol: str, limit: int) -> List[Dict[str, Any]]:
    base_time = datetime.now(timezone.utc)
    items: List[Dict[str, Any]] = []
    for idx in range(limit):
        published = base_time - timedelta(minutes=15 * idx)
        items.append(
            {
                "id": f"mock-{symbol}-{idx}",
                "title": f"Mock news {idx} for {symbol}",
                "published_at": published.isoformat(),
            }
        )
    return items


async def binance_ohlcv(symbol: str, limit: int = 600) -> Dict[str, Any]:
    """Fetch OHLCV candles via market-data adapters with deterministic fallback."""

    try:
        candles = await asyncio.to_thread(fetch_recent_candles, symbol, limit)
        source = "live" if _live_market_data_enabled() else "demo"
    except Exception as exc:  # pragma: no cover - network/adapter issues
        logger.debug("fetch_recent_candles failed for %s: %s", symbol, exc, exc_info=True)
        candles = _fallback_candles(symbol, limit)
        source = "fallback"
    normalised = _normalise_candles(candles)
    return {"symbol": symbol, "source": source, "candles": normalised}


async def twitter_sentiment(symbol: str) -> Dict[str, Any]:
    """Return a light-touch sentiment summary for a symbol."""

    base_symbol = _strip_quote(symbol)
    client = _twitter_client()
    try:
        summary = await asyncio.to_thread(client.sentiment_for_symbol, base_symbol)
    except Exception as exc:  # pragma: no cover - network/adapter issues
        logger.debug("twitter sentiment failed for %s: %s", symbol, exc, exc_info=True)
        summary = {"score": 0.0, "label": "neutral", "source": "error"}
    payload = {"sentiment": summary}
    payload["score"] = summary.get("score", 0.0)
    return payload


async def cryptopanic_news(symbol: str, limit: int = 200) -> Dict[str, Any]:
    """Fetch latest news headlines tagged for a symbol."""

    base_symbol = _strip_quote(symbol)
    client = _cryptopanic_client()
    try:
        items = await asyncio.to_thread(client.fetch_latest, base_symbol, limit)
    except Exception as exc:  # pragma: no cover - network/adapter issues
        logger.debug("cryptopanic fetch failed for %s: %s", symbol, exc, exc_info=True)
        items = _fallback_news(base_symbol, limit)
    return {"symbol": symbol, "news": items[:limit]}


__all__ = ["binance_ohlcv", "twitter_sentiment", "cryptopanic_news"]

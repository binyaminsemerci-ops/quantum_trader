"""Async helpers providing market/sentiment data for training/backtests.

Enhanced version with multi-source data integration from CoinGecko, Fear & Greed Index,
Reddit sentiment, CryptoCompare news, CoinPaprika metrics, and Messari on-chain data.
Routes requests through shared config + adapter layer with deterministic fallbacks.
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
from backend.enhanced_data_feeds import EnhancedDataFeed, get_enhanced_market_data

logger = logging.getLogger(__name__)

_TWITTER_CLIENT: TwitterClient | None = None
_CRYPTOPANIC_CLIENT = None
_ENHANCED_DATA_FEED: EnhancedDataFeed | None = None


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


def _enhanced_feed() -> EnhancedDataFeed:
    global _ENHANCED_DATA_FEED
    if _ENHANCED_DATA_FEED is None:
        _ENHANCED_DATA_FEED = EnhancedDataFeed()
    return _ENHANCED_DATA_FEED


def _cryptopanic_client() -> None:
    """CryptoPanic has been removed as a live data source; return None.

    Keep this function for compatibility but do not instantiate any external
    client or make network calls.
    """
    return None


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
    """Return deterministic fallback news for a symbol.

    CryptoPanic has been removed as an external live news provider. This
    helper now returns deterministic mock news so callers keep working and
    tests remain stable.
    """
    base_symbol = _strip_quote(symbol)
    items = _fallback_news(base_symbol, limit)
    return {"symbol": symbol, "news": items[:limit]}


# Enhanced multi-source data functions
async def enhanced_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Get enhanced market data from multiple free API sources."""
    try:
        enhanced_data = await get_enhanced_market_data(symbols)
        return {
            "symbols": symbols,
            "data": enhanced_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "enhanced_multi_api"
        }
    except Exception as exc:
        logger.debug("Enhanced market data failed: %s", exc, exc_info=True)
        return {
            "symbols": symbols,
            "data": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback"
        }


async def fear_greed_index() -> Dict[str, Any]:
    """Get Fear & Greed Index for market sentiment."""
    try:
        async with EnhancedDataFeed() as feed:
            data = await feed.get_fear_greed_index()
        return data
    except Exception as exc:
        logger.debug("Fear & Greed Index failed: %s", exc, exc_info=True)
        return {
            "current": {"value": 50, "value_classification": "Neutral"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback"
        }


async def reddit_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """Get Reddit sentiment analysis for crypto symbols."""
    try:
        async with EnhancedDataFeed() as feed:
            data = await feed.get_reddit_sentiment(symbols)
        return data
    except Exception as exc:
        logger.debug("Reddit sentiment failed: %s", exc, exc_info=True)
        return {
            "symbols": {symbol: {"sentiment_score": 0.0, "total_posts": 0} for symbol in symbols},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback"
        }


async def comprehensive_crypto_news() -> Dict[str, Any]:
    """Get comprehensive crypto news from multiple sources."""
    try:
        async with EnhancedDataFeed() as feed:
            data = await feed.get_cryptocompare_data([])  # News doesn't need symbols
        return data
    except Exception as exc:
        logger.debug("Comprehensive news failed: %s", exc, exc_info=True)
        return {
            "news": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback"
        }


async def on_chain_metrics(symbols: List[str]) -> Dict[str, Any]:
    """Get on-chain metrics from Messari."""
    try:
        async with EnhancedDataFeed() as feed:
            data = await feed.get_messari_metrics(symbols)
        return data
    except Exception as exc:
        logger.debug("On-chain metrics failed: %s", exc, exc_info=True)
        return {
            "metrics": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback"
        }


async def market_indicators() -> Dict[str, Any]:
    """Get global market indicators."""
    try:
        async with EnhancedDataFeed() as feed:
            data = await feed.get_market_indicators()
        return data
    except Exception as exc:
        logger.debug("Market indicators failed: %s", exc, exc_info=True)
        return {
            "global_stats": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback"
        }


__all__ = [
    "binance_ohlcv",
    "twitter_sentiment",
    "enhanced_market_data",
    "fear_greed_index",
    "reddit_sentiment",
    "comprehensive_crypto_news",
    "on_chain_metrics",
    "market_indicators"
]

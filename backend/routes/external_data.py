from typing import Dict, Any


async def binance_ohlcv(symbol: str, limit: int = 600) -> Dict[str, Any]:
    # minimal stub (tests will monkeypatch this)
    return {'candles': []}


async def twitter_sentiment(symbol: str) -> Dict[str, Any]:
    return {'score': 0.0, 'label': 'neutral', 'source': 'stub'}


async def cryptopanic_news(symbol: str, limit: int = 200) -> Dict[str, Any]:
    return {'news': []}
import asyncio
from typing import Dict, Any


async def binance_ohlcv(symbol: str, limit: int = 600) -> Dict[str, Any]:
    """Return OHLCV candles for a symbol.

    This is a minimal stub used in tests and local development. Production
    code should replace this with a real fetcher from Binance or a cached
    data source.
    """
    # Return a simple synthetic structure matching what tests expect.
    candles: list[Dict[str, Any]] = []
    price: float = 100.0
    for i in range(limit):
        candles.append({
            "timestamp": f"t{i}",
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price + (i % 3 - 1) * 0.1,
            "volume": 100 + i,
        })
        price = float(candles[-1]["close"])

    # mimic I/O latency
    await asyncio.sleep(0)
    return {"candles": candles}


async def twitter_sentiment(symbol: str) -> Dict[str, Any]:
    """Return a minimal sentiment payload for a symbol."""
    await asyncio.sleep(0)
    return {"score": 0.0, "label": "neutral", "source": "stub"}


async def cryptopanic_news(symbol: str, limit: int = 200) -> Dict[str, Any]:
    """Return a minimal news payload for a symbol."""
    await asyncio.sleep(0)
    return {"news": []}

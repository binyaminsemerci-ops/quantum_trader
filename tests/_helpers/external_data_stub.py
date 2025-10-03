from typing import Dict, Any


async def binance_ohlcv(symbol: str, limit: int = 600) -> Dict[str, Any]:
    return {"candles": []}


async def twitter_sentiment(symbol: str) -> Dict[str, Any]:
    return {"score": 0.0, "label": "neutral", "source": "stub"}


async def cryptopanic_news(symbol: str, limit: int = 200) -> Dict[str, Any]:
    return {"news": []}

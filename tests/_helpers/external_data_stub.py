
from typing import Dict, Any

async def binance_ohlcv(symbol: str, limit: int = 600) -> Dict[str, Any]:
    return {"candles": []}

async def twitter_sentiment(symbol: str) -> Dict[str, Any]:
    return {"score": 0.0, "label": "neutral", "source": "stub"}

def cryptopanic_news():
    return [{"headline": "Mock news", "source": "cryptopanic"}]




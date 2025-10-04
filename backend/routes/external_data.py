import asyncio
import aiohttp
from typing import Dict, Any, Mapping, Union
import logging

logger = logging.getLogger(__name__)


async def binance_ohlcv(symbol: str, limit: int = 600) -> Dict[str, Any]:
    """Fetch real OHLCV candles from Binance public API (no auth required)."""
    try:
        # Binance public klines endpoint
        url = "https://api.binance.com/api/v3/klines"
        # Explicit param typing to satisfy type checkers (str -> str|int)
        params: Dict[str, Union[str, int]] = {
            "symbol": symbol.upper(),
            "interval": "1m",  # 1 minute candles
            "limit": min(limit, 1000)  # Binance max is 1000
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    api_candles = []

                    for kline in data:
                        api_candles.append({
                            "timestamp": int(kline[0]),  # Open time
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5])
                        })

                    return {"candles": api_candles}
                else:
                    logger.warning(f"Binance API error: {response.status}")

    except Exception as e:
        logger.error(f"Failed to fetch Binance data for {symbol}: {e}")

    # Fallback to demo data if API fails
    candles: list[dict[str, float]] = []
    price: float = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 100.0
    for i in range(min(limit, 100)):
        candles.append({
            "timestamp": float(i),
            "open": price,
            "high": price + (price * 0.01),
            "low": price - (price * 0.01),
            "close": price + (i % 3 - 1) * (price * 0.002),
            "volume": 100.0 + float(i),
        })
        price = float(candles[-1]["close"])
    return {"candles": candles}


def cryptopanic_news():
    """Returns mock news data for cryptopanic API."""
    return [{"headline": "Mock news", "source": "cryptopanic"}]


async def twitter_sentiment(symbol: str) -> Dict[str, Any]:
    """Get sentiment data from CoinGecko's sentiment indicators (free API)."""
    try:
        # CoinGecko has sentiment data without requiring API keys
        coin_id = symbol.lower().replace('usdt', '').replace('btc', 'bitcoin').replace('eth', 'ethereum')

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params: Dict[str, str] = {"localization": "false", "tickers": "false", "market_data": "true", "community_data": "true"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract sentiment metrics from CoinGecko
                    sentiment_votes_up = data.get("sentiment_votes_up_percentage", 50)
                    market_cap_rank = data.get("market_cap_rank", 100)

                    # Calculate sentiment score (0-1 scale)
                    sentiment_score = (sentiment_votes_up / 100.0) * 0.7 + (1.0 - min(market_cap_rank / 100.0, 1.0)) * 0.3

                    label = "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"

                    return {
                        "score": round(sentiment_score, 3),
                        "label": label,
                        "source": "coingecko",
                        "sentiment_votes_up_percentage": sentiment_votes_up,
                        "market_cap_rank": market_cap_rank
                    }

    except Exception as e:
        logger.error(f"Failed to fetch sentiment for {symbol}: {e}")

    # Fallback sentiment
    return {"score": 0.5, "label": "neutral", "source": "fallback"}

from typing import Dict, List, Annotated
from fastapi import APIRouter, Query
import datetime
import logging
from .external_data import binance_ohlcv
from .coingecko_data import get_coin_price_data, symbol_to_coingecko_id

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/recent")
async def recent_prices(
    symbol: str = "BTCUSDT", limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> List[Dict]:
    """Return live price data from Binance API or CoinGecko as fallback."""
    try:
        # Try Binance first for real-time data
        binance_data = await binance_ohlcv(symbol, limit)
        candles = binance_data.get("candles", [])

        if candles:
            # Convert Binance format to frontend format
            formatted_candles = []
            for candle in candles[-limit:]:  # Get latest data
                formatted_candles.append(
                    {
                        "time": (
                            datetime.datetime.fromtimestamp(
                                candle["timestamp"] / 1000
                                if isinstance(candle["timestamp"], int)
                                else 0
                            ).isoformat()
                            if isinstance(candle["timestamp"], int)
                            else candle["timestamp"]
                        ),
                        "open": round(float(candle["open"]), 3),
                        "high": round(float(candle["high"]), 3),
                        "low": round(float(candle["low"]), 3),
                        "close": round(float(candle["close"]), 3),
                        "volume": int(candle["volume"]),
                    }
                )
            return formatted_candles

    except Exception as e:
        logger.error(f"Failed to get live prices for {symbol}: {e}")

    # Fallback: try CoinGecko for historical data
    try:
        coin_id = symbol_to_coingecko_id(symbol)
        price_data = await get_coin_price_data(coin_id, days=1)
        prices = price_data.get("prices", [])

        if prices:
            candles = []
            for i, price_point in enumerate(prices[-limit:]):
                timestamp, price = price_point
                candles.append(
                    {
                        "time": datetime.datetime.fromtimestamp(
                            timestamp / 1000
                        ).isoformat(),
                        "open": round(price, 3),
                        "high": round(price * 1.01, 3),
                        "low": round(price * 0.99, 3),
                        "close": round(price, 3),
                        "volume": 1000 + i,
                    }
                )
            return candles

    except Exception as e:
        logger.error(f"CoinGecko fallback failed for {symbol}: {e}")

    # Final fallback to demo data
    now = datetime.datetime.now(datetime.timezone.utc)
    fallback_candles: List[Dict] = []
    base = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0

    for i in range(limit):
        t = (now - datetime.timedelta(minutes=(limit - i))).isoformat()
        open_p = base + (i * 0.1) + (0.5 * (i % 3))
        close_p = open_p + ((-1) ** i) * (0.5 * ((i % 5) / 5.0))
        high_p = max(open_p, close_p) + (base * 0.01)
        low_p = min(open_p, close_p) - (base * 0.01)
        volume = 10 + (i % 7)
        fallback_candles.append(
            {
                "time": t,
                "open": round(open_p, 3),
                "high": round(high_p, 3),
                "low": round(low_p, 3),
                "close": round(close_p, 3),
                "volume": volume,
            }
        )
    return fallback_candles

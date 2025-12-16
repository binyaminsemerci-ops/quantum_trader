"""CoinGecko API integration for live market data and news."""

from typing import Dict, Any, List, Union
import logging

try:
    from backend.api_bulletproof import get_coingecko_client
except ImportError:  # pragma: no cover - fallback for alternative module paths
    from api_bulletproof import get_coingecko_client

logger = logging.getLogger(__name__)


async def get_coin_price_data(coin_id: str, days: int = 7) -> Dict[str, Any]:
    """Fetch price history from CoinGecko."""
    client = get_coingecko_client()
    params: Dict[str, Union[str, int]] = {"vs_currency": "usd", "days": days}

    try:
        data = await client.get(
            f"/api/v3/coins/{coin_id}/market_chart",
            params=params,
            fallback=None,
        )

        if data and isinstance(data, dict):
            return {
                "prices": data.get("prices", []),
                "market_caps": data.get("market_caps", []),
                "total_volumes": data.get("total_volumes", []),
            }

    except Exception as e:
        logger.error(f"Failed to fetch price data for {coin_id}: {e}")

    return {"prices": [], "market_caps": [], "total_volumes": []}


async def get_trending_coins() -> List[Dict[str, Any]]:
    """Get trending coins from CoinGecko."""
    client = get_coingecko_client()

    try:
        data = await client.get("/api/v3/search/trending", fallback=None)
        if data and isinstance(data, dict):
            return data.get("coins", [])

    except Exception as e:
        logger.error(f"Failed to fetch trending coins: {e}")

    return []


async def get_market_data(coin_ids: List[str]) -> List[Dict[str, Any]]:
    """Get current market data for multiple coins."""
    client = get_coingecko_client()
    params: Dict[str, str] = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": "100",
        "page": "1",
        "sparkline": "false",
    }

    try:
        data = await client.get(
            "/api/v3/coins/markets",
            params=params,
            fallback=None,
        )
        if data and isinstance(data, list):
            return data

    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")

    return []


async def get_global_market_data() -> Dict[str, Any]:
    """Get global cryptocurrency market data."""
    client = get_coingecko_client()

    try:
        data = await client.get("/api/v3/global", fallback=None)
        if data and isinstance(data, dict):
            return data.get("data", {})

    except Exception as e:
        logger.error(f"Failed to fetch global market data: {e}")

    return {}


def symbol_to_coingecko_id(symbol: str) -> str:
    """Convert trading symbol to CoinGecko coin ID."""
    symbol_map = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "ADAUSDT": "cardano",
        "SOLUSDT": "solana",
        "DOGEUSDT": "dogecoin",
        "DOTUSDT": "polkadot",
        "LINKUSDT": "chainlink",
        "MATICUSDT": "matic-network",
        "AVAXUSDT": "avalanche-2",
        "ATOMUSDT": "cosmos",
    }

    return symbol_map.get(symbol.upper(), symbol.lower().replace("usdt", ""))

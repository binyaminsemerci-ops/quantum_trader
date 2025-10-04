"""CoinGecko API integration for live market data and news."""

import asyncio
import aiohttp
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


async def get_coin_price_data(coin_id: str, days: int = 7) -> Dict[str, Any]:
    """Fetch price history from CoinGecko."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "prices": data.get("prices", []),
                        "market_caps": data.get("market_caps", []),
                        "total_volumes": data.get("total_volumes", [])
                    }
    except Exception as e:
        logger.error(f"Failed to fetch price data for {coin_id}: {e}")
    
    return {"prices": [], "market_caps": [], "total_volumes": []}


async def get_trending_coins() -> List[Dict[str, Any]]:
    """Get trending coins from CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("coins", [])
    except Exception as e:
        logger.error(f"Failed to fetch trending coins: {e}")
    
    return []


async def get_market_data(coin_ids: List[str]) -> List[Dict[str, Any]]:
    """Get current market data for multiple coins."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
    
    return []


async def get_global_market_data() -> Dict[str, Any]:
    """Get global cryptocurrency market data."""
    try:
        url = "https://api.coingecko.com/api/v3/global"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
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
        "ATOMUSDT": "cosmos"
    }
    
    return symbol_map.get(symbol.upper(), symbol.lower().replace("usdt", ""))
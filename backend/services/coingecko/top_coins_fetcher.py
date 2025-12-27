"""
CoinGecko Top Coins Fetcher
Fetches top 100 coins by 24h volume, filtered for main base + Layer 1 + Layer 2.
Updates daily for dynamic coin selection.
"""

import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import aiohttp
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TopCoinsFetcher:
    """
    Fetches top coins from CoinGecko API based on 24h trading volume.
    Filters for main base coins + Layer 1 + Layer 2 projects.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Main base coins (always include)
    MAIN_BASE_COINS = [
        "bitcoin", "ethereum", "binancecoin", "ripple", "cardano",
        "solana", "polkadot", "dogecoin", "tron", "avalanche-2",
        "polygon", "litecoin", "shiba-inu", "uniswap", "chainlink"
    ]
    
    # Layer 1 categories on CoinGecko
    LAYER1_CATEGORIES = [
        "layer-1", "smart-contract-platform", "proof-of-stake",
        "cosmos-ecosystem", "polkadot-ecosystem", "ethereum-ecosystem"
    ]
    
    # Layer 2 categories on CoinGecko
    LAYER2_CATEGORIES = [
        "layer-2", "ethereum-layer-2", "arbitrum-ecosystem",
        "optimism-ecosystem", "polygon-ecosystem", "zksync-ecosystem"
    ]
    
    def __init__(self, cache_dir: str = "backend/data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "top_coins_cache.json"
        self.cache_ttl_hours = 24  # Refresh daily
        
    async def fetch_top_coins(
        self, 
        limit: int = 100,
        force_refresh: bool = False
    ) -> List[Dict]:
        """
        Fetch top coins by 24h volume, filtered for main base + L1 + L2.
        
        Args:
            limit: Number of coins to return (default 100)
            force_refresh: Force API call even if cache is valid
            
        Returns:
            List of coin dictionaries with symbol, name, volume, market_cap, etc.
        """
        # Check cache first
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                logger.info(f"Using cached top {limit} coins")
                return cached[:limit]
        
        logger.info(f"Fetching top {limit} coins from CoinGecko...")
        
        try:
            # Fetch top coins by volume
            coins = await self._fetch_coins_by_volume(limit * 3)  # Get more to filter
            
            # Filter for main base + L1 + L2
            filtered = self._filter_coins(coins, limit)
            
            # Save to cache
            self._save_cache(filtered)
            
            logger.info(f"Successfully fetched {len(filtered)} top coins")
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to fetch top coins: {e}")
            # Try to return stale cache on error
            cached = self._load_cache()
            if cached:
                logger.warning("Using stale cache due to fetch error")
                return cached[:limit]
            raise
    
    async def _fetch_coins_by_volume(self, limit: int) -> List[Dict]:
        """
        Fetch coins sorted by 24h volume from CoinGecko.
        """
        url = f"{self.BASE_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "volume_desc",  # Sort by 24h volume descending
            "per_page": min(limit, 250),  # CoinGecko max per page
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"CoinGecko API error: {response.status}")
                
                data = await response.json()
                
                # Transform to our format
                coins = []
                for coin in data:
                    coins.append({
                        "id": coin.get("id"),
                        "symbol": coin.get("symbol", "").upper(),
                        "name": coin.get("name"),
                        "market_cap_rank": coin.get("market_cap_rank"),
                        "market_cap": coin.get("market_cap", 0),
                        "volume_24h": coin.get("total_volume", 0),
                        "price": coin.get("current_price", 0),
                        "price_change_24h": coin.get("price_change_percentage_24h", 0),
                        "categories": [],  # Will be populated if needed
                        "binance_symbol": self._to_binance_symbol(coin.get("symbol", ""))
                    })
                
                return coins
    
    def _filter_coins(self, coins: List[Dict], limit: int) -> List[Dict]:
        """
        Filter coins to include main base + L1 + L2 projects.
        Prioritize by volume within each category.
        """
        filtered = []
        
        # Always include main base coins (by ID)
        for coin in coins:
            if coin["id"] in self.MAIN_BASE_COINS:
                coin["category"] = "main_base"
                filtered.append(coin)
        
        # Add high volume coins as L1/L2 (simplified - in production would check categories)
        # For now, take top volume coins that aren't meme coins
        exclude_meme = ["shib", "doge", "pepe", "floki", "bonk"]
        
        for coin in coins:
            if len(filtered) >= limit:
                break
                
            symbol_lower = coin["symbol"].lower()
            
            # Skip if already included
            if any(c["symbol"] == coin["symbol"] for c in filtered):
                continue
            
            # Skip obvious meme coins (but keep if main base)
            if symbol_lower in exclude_meme and coin["id"] not in self.MAIN_BASE_COINS:
                continue
            
            # Add high volume coins
            coin["category"] = "layer1_layer2"
            filtered.append(coin)
        
        # Sort by volume and take top limit
        filtered.sort(key=lambda x: x["volume_24h"], reverse=True)
        return filtered[:limit]
    
    def _to_binance_symbol(self, symbol: str) -> str:
        """
        Convert CoinGecko symbol to Binance USDM perpetual symbol.
        """
        symbol = symbol.upper()
        
        # Special mappings
        mappings = {
            "MIOTA": "IOTAUSDT",
            "BCH": "BCHUSDT",
            "BSV": "BSVUSDT",
        }
        
        if symbol in mappings:
            return mappings[symbol]
        
        return f"{symbol}USDT"
    
    def _load_cache(self) -> Optional[List[Dict]]:
        """
        Load coins from cache if fresh enough.
        """
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            cached_at = datetime.fromisoformat(cache["cached_at"])
            age = datetime.now() - cached_at
            
            if age < timedelta(hours=self.cache_ttl_hours):
                logger.info(f"Cache age: {age.seconds // 3600}h (fresh)")
                return cache["coins"]
            else:
                logger.info(f"Cache age: {age.seconds // 3600}h (stale)")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_cache(self, coins: List[Dict]):
        """
        Save coins to cache with timestamp.
        """
        try:
            cache = {
                "cached_at": datetime.now().isoformat(),
                "coins": coins
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(coins)} coins to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def get_binance_symbols(self, limit: int = 100) -> List[str]:
        """
        Get list of Binance perpetual symbols from top coins.
        
        Returns:
            List of symbols like ["BTCUSDT", "ETHUSDT", ...]
        """
        coins = await self.fetch_top_coins(limit)
        return [coin["binance_symbol"] for coin in coins]


async def main():
    """
    Test the fetcher.
    """
    logging.basicConfig(level=logging.INFO)
    
    fetcher = TopCoinsFetcher()
    
    # Fetch top 100
    coins = await fetcher.fetch_top_coins(100, force_refresh=True)
    
    print(f"\nTop 100 coins by 24h volume:")
    print("=" * 80)
    for i, coin in enumerate(coins[:20], 1):
        print(f"{i:3d}. {coin['symbol']:8s} | "
              f"Vol: ${coin['volume_24h']:12,.0f} | "
              f"MCap: ${coin['market_cap']:12,.0f} | "
              f"Binance: {coin['binance_symbol']}")
    
    print(f"\n... and {len(coins) - 20} more")
    
    # Get Binance symbols
    symbols = await fetcher.get_binance_symbols(100)
    print(f"\nBinance symbols ({len(symbols)}):")
    print(", ".join(symbols[:30]))
    print(f"... and {len(symbols) - 30} more")


if __name__ == "__main__":
    asyncio.run(main())

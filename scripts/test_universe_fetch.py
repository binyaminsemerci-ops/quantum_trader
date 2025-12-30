#!/usr/bin/env python3
"""Test universe fetching"""
import asyncio
from backend.services.coingecko.top_coins_fetcher import TopCoinsFetcher

async def main():
    fetcher = TopCoinsFetcher()
    coins = await fetcher.fetch_top_coins(20, force_refresh=True)
    
    print(f"\n{'='*80}")
    print(f"TOP 20 COINS BY 24H VOLUME (Filtered: Main Base + L1 + L2)")
    print(f"{'='*80}")
    
    for i, coin in enumerate(coins, 1):
        symbol = coin["symbol"].upper()
        name = coin["name"]
        volume_b = coin["volume_24h"] / 1e9
        category = coin.get("category", "unknown")
        print(f"{i:2d}. {symbol:10s} {name:20s} ${volume_b:6.2f}B  [{category}]")

if __name__ == "__main__":
    asyncio.run(main())

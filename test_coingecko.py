"""Test CoinGecko fetcher"""
import asyncio
import logging
from backend.services.coingecko.top_coins_fetcher import TopCoinsFetcher

logging.basicConfig(level=logging.INFO)

async def main():
    fetcher = TopCoinsFetcher()
    
    print("Fetching top 100 coins from CoinGecko...")
    coins = await fetcher.fetch_top_coins(100, force_refresh=True)
    
    print(f"\nFetched {len(coins)} coins")
    print("=" * 80)
    
    for i, coin in enumerate(coins[:20], 1):
        print(f"{i:3d}. {coin['symbol']:8s} | Vol: ${coin['volume_24h']:12,.0f} | {coin['binance_symbol']}")
    
    print(f"\n... and {len(coins) - 20} more coins")
    
    # Get Binance symbols
    symbols = await fetcher.get_binance_symbols(100)
    print(f"\nBinance symbols: {', '.join(symbols[:30])}")

if __name__ == "__main__":
    asyncio.run(main())

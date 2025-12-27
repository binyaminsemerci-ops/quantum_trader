"""
Multi-Source Data Collector
Collects historical data for all symbols in dynamic universe from multiple sources:
- Binance (primary)
- CoinGecko (price + market data)
- Future: Add more exchanges as needed

Runs daily to refresh data for model training.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

from backend.services.universe_manager import get_universe_manager
from backend.domains.learning.data_pipeline import HistoricalDataFetcher, FeatureConfig

logger = logging.getLogger(__name__)


class MultiSourceDataCollector:
    """
    Collects data from multiple sources for all universe symbols.
    """
    
    def __init__(
        self,
        data_dir: str = "backend/data/market_data",
        lookback_days: int = 90,
        timeframe: str = "5m"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.lookback_days = lookback_days
        self.timeframe = timeframe
        
        # Data fetchers
        self.binance_fetcher = HistoricalDataFetcher(db_session=None, use_testnet=False)
        
    async def collect_all_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all symbols in universe.
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        # Get current universe
        universe_manager = get_universe_manager()
        symbols = universe_manager.get_symbols()
        
        logger.info(f"[DATA COLLECTION] Collecting data for {len(symbols)} symbols")
        logger.info(f"[DATA COLLECTION] Lookback: {self.lookback_days} days, Timeframe: {self.timeframe}")
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Collect from Binance (primary source)
        logger.info("[DATA COLLECTION] Fetching from Binance...")
        binance_data = await self._collect_from_binance(
            symbols, start_date, end_date, force_refresh
        )
        
        # TODO: Add CoinGecko data collection
        # coingecko_data = await self._collect_from_coingecko(symbols)
        
        # Merge data sources
        merged_data = self._merge_data_sources(binance_data, {})
        
        # Save to disk
        await self._save_data(merged_data)
        
        logger.info(
            f"[DATA COLLECTION] Collected data for {len(merged_data)} symbols, "
            f"Total rows: {sum(len(df) for df in merged_data.values())}"
        )
        
        return merged_data
    
    async def _collect_from_binance(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect OHLCV data from Binance.
        """
        # Check if we have cached data
        if not force_refresh:
            cached = self._load_cached_data(symbols)
            if cached:
                logger.info(f"[BINANCE] Using cached data for {len(cached)} symbols")
                
                # Only fetch symbols not in cache
                cached_symbols = set(cached.keys())
                missing_symbols = [s for s in symbols if s not in cached_symbols]
                
                if missing_symbols:
                    logger.info(f"[BINANCE] Fetching {len(missing_symbols)} new symbols")
                    new_data = await self.binance_fetcher.fetch_historical_data(
                        missing_symbols, start_date, end_date, self.timeframe
                    )
                    
                    # Split by symbol
                    for symbol in missing_symbols:
                        symbol_data = new_data[new_data['symbol'] == symbol]
                        if not symbol_data.empty:
                            cached[symbol] = symbol_data
                
                return cached
        
        # Fetch all data from Binance
        try:
            df = await self.binance_fetcher.fetch_historical_data(
                symbols, start_date, end_date, self.timeframe
            )
            
            # Split by symbol
            result = {}
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol]
                if not symbol_data.empty:
                    result[symbol] = symbol_data
                else:
                    logger.warning(f"[BINANCE] No data for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"[BINANCE] Failed to fetch data: {e}")
            return {}
    
    def _merge_data_sources(
        self,
        binance_data: Dict[str, pd.DataFrame],
        coingecko_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Merge data from multiple sources.
        Currently just returns Binance data, but can be extended to merge with CoinGecko.
        """
        # For now, just use Binance data
        # TODO: Merge with CoinGecko market data (volume, market cap, etc.)
        return binance_data
    
    def _load_cached_data(self, symbols: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load cached data from disk if available and fresh.
        """
        cache_file = self.data_dir / f"collected_data_{self.timeframe}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            # Load all cached data
            df = pd.read_parquet(cache_file)
            
            # Check age
            if 'timestamp' in df.columns:
                latest_timestamp = df['timestamp'].max()
                age_hours = (datetime.utcnow() - latest_timestamp).total_seconds() / 3600
                
                if age_hours > 24:  # Older than 24 hours
                    logger.info(f"[CACHE] Data is {age_hours:.1f}h old, refreshing")
                    return None
            
            # Split by symbol
            result = {}
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol]
                if not symbol_data.empty:
                    result[symbol] = symbol_data
            
            return result if result else None
            
        except Exception as e:
            logger.warning(f"[CACHE] Failed to load cached data: {e}")
            return None
    
    async def _save_data(self, data: Dict[str, pd.DataFrame]):
        """
        Save collected data to disk.
        """
        if not data:
            logger.warning("[SAVE] No data to save")
            return
        
        try:
            # Combine all symbols into single DataFrame
            df = pd.concat(data.values(), ignore_index=True)
            
            # Save as parquet (efficient compression)
            cache_file = self.data_dir / f"collected_data_{self.timeframe}.parquet"
            df.to_parquet(cache_file, index=False)
            
            logger.info(f"[SAVE] Saved {len(df)} rows to {cache_file}")
            
            # Also save metadata
            metadata = {
                "symbols_count": len(data),
                "total_rows": len(df),
                "timeframe": self.timeframe,
                "start_date": df['timestamp'].min().isoformat() if 'timestamp' in df else None,
                "end_date": df['timestamp'].max().isoformat() if 'timestamp' in df else None,
                "collected_at": datetime.utcnow().isoformat()
            }
            
            import json
            meta_file = self.data_dir / f"collected_data_{self.timeframe}_meta.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"[SAVE] Failed to save data: {e}")
    
    async def run_daily_collection(self):
        """
        Background task to collect data daily.
        """
        logger.info("[COLLECTOR] Starting daily data collection loop...")
        
        while True:
            try:
                # Collect data
                logger.info("[COLLECTOR] Starting daily data collection...")
                await self.collect_all_data(force_refresh=False)
                logger.info("[COLLECTOR] Daily collection complete")
                
                # Wait 24 hours
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"[COLLECTOR] Error in daily collection: {e}")
                # Wait 1 hour before retry on error
                await asyncio.sleep(3600)


# Global instance
_data_collector: MultiSourceDataCollector = None


def get_data_collector() -> MultiSourceDataCollector:
    """
    Get global data collector instance.
    """
    global _data_collector
    if _data_collector is None:
        _data_collector = MultiSourceDataCollector()
    return _data_collector


async def main():
    """
    Test data collection.
    """
    logging.basicConfig(level=logging.INFO)
    
    collector = MultiSourceDataCollector(lookback_days=30)
    
    # Collect data
    data = await collector.collect_all_data(force_refresh=True)
    
    print(f"\nCollected data for {len(data)} symbols")
    
    # Show sample
    for symbol, df in list(data.items())[:5]:
        print(f"\n{symbol}: {len(df)} rows")
        print(df.head())


if __name__ == "__main__":
    asyncio.run(main())

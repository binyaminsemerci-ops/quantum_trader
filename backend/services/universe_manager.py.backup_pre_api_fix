"""
Universe Manager - Dynamically manages trading universe based on top coins by volume.
Refreshes daily and integrates with data fetching pipeline.
"""

import asyncio
import logging
from typing import List, Set
from datetime import datetime, timedelta
from pathlib import Path
import json

from backend.services.coingecko.top_coins_fetcher import TopCoinsFetcher

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Manages the dynamic trading universe.
    - Fetches top coins daily from CoinGecko
    - Updates symbol list for data fetching
    - Triggers historical data download for new symbols
    """
    
    def __init__(
        self,
        max_symbols: int = 100,
        refresh_interval_hours: int = 24,
        data_dir: str = "backend/data"
    ):
        self.max_symbols = max_symbols
        self.refresh_interval_hours = refresh_interval_hours
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.universe_file = self.data_dir / "universe.json"
        self.fetcher = TopCoinsFetcher()
        
        self._current_symbols: Set[str] = set()
        self._last_refresh: datetime = None
        
    async def initialize(self):
        """
        Initialize universe on startup.
        Loads from cache or fetches from CoinGecko.
        """
        logger.info("Initializing universe manager...")
        
        # Try to load existing universe
        loaded = self._load_universe()
        
        # Check if refresh needed
        if not loaded or self._needs_refresh():
            logger.info("Refreshing universe from CoinGecko...")
            await self.refresh_universe()
        else:
            logger.info(f"Loaded {len(self._current_symbols)} symbols from cache")
    
    async def refresh_universe(self, force: bool = False):
        """
        Refresh universe from CoinGecko.
        
        Args:
            force: Force refresh even if not due
        """
        if not force and not self._needs_refresh():
            logger.debug("Universe refresh not needed yet")
            return
        
        try:
            logger.info(f"Fetching top {self.max_symbols} coins...")
            
            # Fetch top coins
            symbols = await self.fetcher.get_binance_symbols(self.max_symbols)
            
            # Detect changes
            old_symbols = self._current_symbols.copy()
            new_symbols = set(symbols)
            
            added = new_symbols - old_symbols
            removed = old_symbols - new_symbols
            
            if added:
                logger.info(f"Added {len(added)} new symbols: {', '.join(list(added)[:10])}...")
            if removed:
                logger.info(f"Removed {len(removed)} symbols: {', '.join(list(removed)[:10])}...")
            
            # Update current universe
            self._current_symbols = new_symbols
            self._last_refresh = datetime.now()
            
            # Save to file
            self._save_universe(symbols)
            
            logger.info(f"Universe refreshed: {len(symbols)} symbols active")
            
            return {
                "total": len(symbols),
                "added": list(added),
                "removed": list(removed)
            }
            
        except Exception as e:
            logger.error(f"Failed to refresh universe: {e}")
            raise
    
    def _needs_refresh(self) -> bool:
        """
        Check if universe needs refresh.
        """
        if self._last_refresh is None:
            return True
        
        age = datetime.now() - self._last_refresh
        return age >= timedelta(hours=self.refresh_interval_hours)
    
    def _load_universe(self) -> bool:
        """
        Load universe from file.
        Returns True if loaded successfully.
        """
        if not self.universe_file.exists():
            return False
        
        try:
            with open(self.universe_file, 'r') as f:
                data = json.load(f)
            
            self._current_symbols = set(data["symbols"])
            self._last_refresh = datetime.fromisoformat(data["last_refresh"])
            
            logger.info(f"Loaded universe: {len(self._current_symbols)} symbols")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load universe: {e}")
            return False
    
    def _save_universe(self, symbols: List[str]):
        """
        Save universe to file.
        """
        try:
            data = {
                "symbols": symbols,
                "last_refresh": self._last_refresh.isoformat(),
                "count": len(symbols)
            }
            
            with open(self.universe_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved universe to {self.universe_file}")
            
        except Exception as e:
            logger.error(f"Failed to save universe: {e}")
    
    def get_symbols(self) -> List[str]:
        """
        Get current list of symbols.
        """
        return sorted(list(self._current_symbols))
    
    def get_universe_info(self) -> dict:
        """
        Get info about current universe.
        """
        return {
            "symbol_count": len(self._current_symbols),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "age_hours": (datetime.now() - self._last_refresh).total_seconds() / 3600 if self._last_refresh else None,
            "next_refresh_hours": self.refresh_interval_hours - ((datetime.now() - self._last_refresh).total_seconds() / 3600) if self._last_refresh else 0,
            "symbols": self.get_symbols()[:20]  # Show first 20
        }
    
    async def run_refresh_loop(self):
        """
        Background task to refresh universe periodically.
        """
        logger.info("Starting universe refresh loop...")
        
        while True:
            try:
                # Wait until next refresh is needed
                if self._last_refresh:
                    age = datetime.now() - self._last_refresh
                    wait_hours = self.refresh_interval_hours - (age.total_seconds() / 3600)
                    
                    if wait_hours > 0:
                        logger.info(f"Next universe refresh in {wait_hours:.1f} hours")
                        await asyncio.sleep(wait_hours * 3600)
                
                # Refresh
                await self.refresh_universe()
                
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                # Wait 1 hour before retry on error
                await asyncio.sleep(3600)


# Global instance
_universe_manager: UniverseManager = None


def get_universe_manager() -> UniverseManager:
    """
    Get global universe manager instance.
    """
    global _universe_manager
    if _universe_manager is None:
        _universe_manager = UniverseManager()
    return _universe_manager


async def main():
    """
    Test the universe manager.
    """
    logging.basicConfig(level=logging.INFO)
    
    manager = UniverseManager(max_symbols=100)
    await manager.initialize()
    
    # Force refresh
    result = await manager.refresh_universe(force=True)
    
    print(f"\nRefresh result:")
    print(f"Total symbols: {result['total']}")
    print(f"Added: {len(result['added'])}")
    print(f"Removed: {len(result['removed'])}")
    
    # Get info
    info = manager.get_universe_info()
    print(f"\nUniverse info:")
    print(f"Symbol count: {info['symbol_count']}")
    print(f"Age: {info['age_hours']:.1f} hours")
    print(f"Next refresh: {info['next_refresh_hours']:.1f} hours")
    print(f"\nTop 20 symbols:")
    print(", ".join(info['symbols']))


if __name__ == "__main__":
    asyncio.run(main())

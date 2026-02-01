#!/usr/bin/env python3
"""
Quantum Trading Universe Service (P0)

Fetches Binance Futures exchangeInfo and publishes allowed symbol set to Redis.
Single source of truth for all gates to consume.

FAIL-CLOSED: On fetch failure, preserves last_ok and marks stale=1.
READ-ONLY: Does NOT place orders or create trading plans.
"""

import os
import sys
import time
import json
import re
import logging
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    # Fallback to urllib if requests not available
    import urllib.request
    import urllib.error
    requests = None

import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration from environment variables"""
    def __init__(self):
        self.UNIVERSE_MODE = os.getenv('UNIVERSE_MODE', 'testnet').lower()
        self.UNIVERSE_REFRESH_SEC = int(os.getenv('UNIVERSE_REFRESH_SEC', '60'))
        self.UNIVERSE_MAX = int(os.getenv('UNIVERSE_MAX', '800'))
        self.REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', '0'))
        self.HTTP_TIMEOUT_SEC = int(os.getenv('HTTP_TIMEOUT_SEC', '10'))
        
        # Binance endpoints
        self.EXCHANGE_INFO_URL = (
            'https://testnet.binancefuture.com/fapi/v1/exchangeInfo'
            if self.UNIVERSE_MODE == 'testnet'
            else 'https://fapi.binance.com/fapi/v1/exchangeInfo'
        )
    
    def validate(self):
        """Validate configuration"""
        if self.UNIVERSE_MODE not in ('testnet', 'mainnet'):
            raise ValueError(f"UNIVERSE_MODE must be testnet or mainnet, got: {self.UNIVERSE_MODE}")
        if self.UNIVERSE_REFRESH_SEC < 10:
            raise ValueError(f"UNIVERSE_REFRESH_SEC must be >= 10, got: {self.UNIVERSE_REFRESH_SEC}")
        if self.UNIVERSE_MAX < 1 or self.UNIVERSE_MAX > 2000:
            raise ValueError(f"UNIVERSE_MAX must be 1-2000, got: {self.UNIVERSE_MAX}")
        logger.info(f"Config validated: mode={self.UNIVERSE_MODE}, refresh={self.UNIVERSE_REFRESH_SEC}s, max={self.UNIVERSE_MAX}")


class UniverseService:
    """Universe Service - fetches and publishes tradeable symbols"""
    
    # Redis keys
    KEY_ACTIVE = 'quantum:cfg:universe:active'
    KEY_LAST_OK = 'quantum:cfg:universe:last_ok'
    KEY_META = 'quantum:cfg:universe:meta'
    
    # Symbol validation regex
    SYMBOL_REGEX = re.compile(r'^[A-Z0-9]{3,20}USDT$')
    
    def __init__(self, config: Config):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        logger.info(f"Connected to Redis: {config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}")
    
    def fetch_exchange_info(self) -> Optional[Dict]:
        """Fetch exchangeInfo from Binance API"""
        try:
            if requests:
                # Use requests library
                response = requests.get(
                    self.config.EXCHANGE_INFO_URL,
                    timeout=self.config.HTTP_TIMEOUT_SEC
                )
                response.raise_for_status()
                return response.json()
            else:
                # Fallback to urllib
                req = urllib.request.Request(self.config.EXCHANGE_INFO_URL)
                with urllib.request.urlopen(req, timeout=self.config.HTTP_TIMEOUT_SEC) as response:
                    data = response.read()
                    return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to fetch exchangeInfo: {e}")
            return None
    
    def filter_symbols(self, exchange_info: Dict) -> List[str]:
        """
        Filter symbols from exchangeInfo
        Criteria: contractType=PERPETUAL, status=TRADING
        """
        symbols = []
        for symbol_info in exchange_info.get('symbols', []):
            if (symbol_info.get('contractType') == 'PERPETUAL' and
                symbol_info.get('status') == 'TRADING'):
                symbol = symbol_info.get('symbol', '')
                if symbol:
                    symbols.append(symbol)
        return symbols
    
    def validate_symbols(self, symbols: List[str]) -> bool:
        """
        Validate symbol list
        - Non-empty
        - All match regex ^[A-Z0-9]{3,20}USDT$
        - Count <= UNIVERSE_MAX
        """
        if not symbols:
            logger.error("Symbol list is empty")
            return False
        
        if len(symbols) > self.config.UNIVERSE_MAX:
            logger.warning(f"Symbol count {len(symbols)} exceeds UNIVERSE_MAX {self.config.UNIVERSE_MAX}, truncating")
            # Note: We'll truncate in publish, not reject
        
        invalid = [s for s in symbols if not self.SYMBOL_REGEX.match(s)]
        if invalid:
            logger.error(f"Invalid symbols found (first 10): {invalid[:10]}")
            return False
        
        return True
    
    def build_universe_json(self, symbols: List[str]) -> str:
        """Build universe JSON document"""
        # Cap to UNIVERSE_MAX
        if len(symbols) > self.config.UNIVERSE_MAX:
            symbols = symbols[:self.config.UNIVERSE_MAX]
        
        universe = {
            'asof_epoch': int(time.time()),
            'source': 'binance_futures_exchangeInfo',
            'mode': self.config.UNIVERSE_MODE,
            'symbols': sorted(symbols),  # Sort for consistency
            'filters': {
                'contractType': 'PERPETUAL',
                'status': 'TRADING'
            }
        }
        return json.dumps(universe)
    
    def publish_universe(self, universe_json: str):
        """
        Publish universe to Redis
        Updates: active, last_ok, meta
        """
        universe = json.loads(universe_json)
        asof = universe['asof_epoch']
        count = len(universe['symbols'])
        
        # Update active
        self.redis_client.set(self.KEY_ACTIVE, universe_json)
        
        # Update last_ok (successful fetch)
        self.redis_client.set(self.KEY_LAST_OK, universe_json)
        
        # Update meta
        self.redis_client.hset(self.KEY_META, 'asof_epoch', asof)
        self.redis_client.hset(self.KEY_META, 'last_ok_epoch', asof)
        self.redis_client.hset(self.KEY_META, 'count', count)
        self.redis_client.hset(self.KEY_META, 'stale', 0)
        self.redis_client.hset(self.KEY_META, 'error', '')
        
        logger.info(f"Published universe: {count} symbols, asof={asof}")
    
    def mark_stale(self, error_msg: str):
        """
        Mark universe as stale on fetch failure
        Preserves last_ok, updates meta.stale=1 and meta.error
        """
        self.redis_client.hset(self.KEY_META, 'stale', 1)
        self.redis_client.hset(self.KEY_META, 'error', error_msg[:500])  # Trim error
        logger.warning(f"Universe marked stale: {error_msg}")
    
    def bootstrap_from_last_ok(self):
        """
        On boot: if active missing but last_ok exists, copy last_ok → active
        Mark stale=1 until first successful fetch
        """
        active_exists = self.redis_client.exists(self.KEY_ACTIVE)
        last_ok_exists = self.redis_client.exists(self.KEY_LAST_OK)
        
        if not active_exists and last_ok_exists:
            logger.info("Active key missing, bootstrapping from last_ok")
            last_ok_json = self.redis_client.get(self.KEY_LAST_OK)
            self.redis_client.set(self.KEY_ACTIVE, last_ok_json)
            
            last_ok = json.loads(last_ok_json)
            self.redis_client.hset(self.KEY_META, 'asof_epoch', last_ok['asof_epoch'])
            self.redis_client.hset(self.KEY_META, 'last_ok_epoch', last_ok['asof_epoch'])
            self.redis_client.hset(self.KEY_META, 'count', len(last_ok['symbols']))
            self.redis_client.hset(self.KEY_META, 'stale', 1)
            self.redis_client.hset(self.KEY_META, 'error', 'bootstrapped_from_last_ok')
            logger.info(f"Bootstrapped {len(last_ok['symbols'])} symbols from last_ok (marked stale)")
        elif not active_exists and not last_ok_exists:
            logger.warning("Neither active nor last_ok exists - first boot, will populate on first fetch")
    
    def refresh_universe(self):
        """Main refresh logic - fetch, validate, publish"""
        logger.info(f"Fetching universe from {self.config.EXCHANGE_INFO_URL}")
        
        # Fetch
        exchange_info = self.fetch_exchange_info()
        if exchange_info is None:
            self.mark_stale('fetch_failed')
            return
        
        # Filter
        symbols = self.filter_symbols(exchange_info)
        logger.info(f"Filtered {len(symbols)} PERPETUAL TRADING symbols")
        
        # Validate
        if not self.validate_symbols(symbols):
            self.mark_stale('validation_failed')
            return
        
        # Build JSON
        universe_json = self.build_universe_json(symbols)
        
        # Publish
        self.publish_universe(universe_json)
        logger.info(f"Universe refresh complete: {len(json.loads(universe_json)['symbols'])} symbols active")
    
    def run(self):
        """Main service loop"""
        logger.info(f"Universe Service starting: mode={self.config.UNIVERSE_MODE}, refresh={self.config.UNIVERSE_REFRESH_SEC}s")
        
        # Bootstrap from last_ok if needed
        self.bootstrap_from_last_ok()
        
        # Immediate first refresh
        try:
            self.refresh_universe()
        except Exception as e:
            logger.error(f"Initial refresh failed: {e}", exc_info=True)
            self.mark_stale(f'initial_refresh_exception:{e}')
        
        # Main loop
        while True:
            try:
                time.sleep(self.config.UNIVERSE_REFRESH_SEC)
                self.refresh_universe()
            except Exception as e:
                logger.error(f"Refresh loop error: {e}", exc_info=True)
                self.mark_stale(f'loop_exception:{e}')


def main():
    """Entry point"""
    try:
        # Load and validate config
        config = Config()
        config.validate()
        
        # Create and run service
        service = UniverseService(config)
        service.run()
        
    except KeyboardInterrupt:
        logger.info("Universe Service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Universe Service fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

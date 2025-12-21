"""
Exchange Data Collector
Fetch OHLC, Funding Rate, and Open Interest data from Binance, Bybit, and Coinbase
All endpoints are public (no API keys required)
"""
import asyncio
import aiohttp
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exchange REST Endpoints
REST_BINANCE = {
    "ohlc": "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    "funding": "https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}",
    "oi": "https://fapi.binance.com/fapi/v1/openInterestHist?symbol={symbol}&period=1h&limit={limit}"
}

REST_BYBIT = {
    "ohlc": "https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=60&limit={limit}",
    "funding": "https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit={limit}"
}

REST_COINBASE = "https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=60"

# Supported symbols
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Cache directory for last timestamps
CACHE_DIR = "/tmp"


class ExchangeDataCollector:
    """Unified data collector for multiple exchanges"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_last_timestamp(self, exchange: str, symbol: str) -> Optional[int]:
        """Get last fetched timestamp from cache"""
        cache_file = os.path.join(CACHE_DIR, f"last_{exchange}_{symbol}.txt")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return int(f.read().strip())
        except Exception as e:
            logger.warning(f"Failed to read cache for {exchange}:{symbol}: {e}")
        return None
    
    def _save_last_timestamp(self, exchange: str, symbol: str, timestamp: int):
        """Save last fetched timestamp to cache"""
        cache_file = os.path.join(CACHE_DIR, f"last_{exchange}_{symbol}.txt")
        try:
            with open(cache_file, 'w') as f:
                f.write(str(timestamp))
        except Exception as e:
            logger.warning(f"Failed to save cache for {exchange}:{symbol}: {e}")
    
    async def _fetch_json(self, url: str, timeout: int = 10) -> Optional[dict]:
        """Fetch JSON data from URL"""
        try:
            async with self.session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    async def fetch_binance_ohlc(self, symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLC data from Binance"""
        url = REST_BINANCE["ohlc"].format(symbol=symbol, interval=interval, limit=limit)
        data = await self._fetch_json(url)
        
        if not data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_numeric(df['timestamp']) // 1000  # Convert to seconds
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['exchange'] = 'binance'
            df['symbol'] = symbol
            
            # Filter out duplicates
            last_ts = self._get_last_timestamp('binance', symbol)
            if last_ts:
                df = df[df['timestamp'] > last_ts]
            
            if not df.empty:
                self._save_last_timestamp('binance', symbol, int(df['timestamp'].max()))
            
            logger.info(f"Binance {symbol}: Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse Binance OHLC for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_bybit_ohlc(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLC data from Bybit"""
        url = REST_BYBIT["ohlc"].format(symbol=symbol, limit=limit)
        data = await self._fetch_json(url)
        
        if not data or data.get('retCode') != 0:
            logger.error(f"Bybit API error for {symbol}: {data}")
            return pd.DataFrame()
        
        try:
            result = data.get('result', {})
            klines = result.get('list', [])
            
            if not klines:
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_numeric(df['timestamp']) // 1000  # Convert to seconds
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['exchange'] = 'bybit'
            df['symbol'] = symbol
            
            # Filter out duplicates
            last_ts = self._get_last_timestamp('bybit', symbol)
            if last_ts:
                df = df[df['timestamp'] > last_ts]
            
            if not df.empty:
                self._save_last_timestamp('bybit', symbol, int(df['timestamp'].max()))
            
            logger.info(f"Bybit {symbol}: Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse Bybit OHLC for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_coinbase_ohlc(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLC data from Coinbase"""
        # Coinbase uses different symbol format (BTC-USD instead of BTCUSDT)
        cb_symbol = symbol.replace('USDT', '-USD')
        url = REST_COINBASE.format(symbol=cb_symbol)
        data = await self._fetch_json(url)
        
        if not data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['exchange'] = 'coinbase'
            df['symbol'] = symbol
            
            # Filter out duplicates
            last_ts = self._get_last_timestamp('coinbase', symbol)
            if last_ts:
                df = df[df['timestamp'] > last_ts]
            
            if not df.empty:
                self._save_last_timestamp('coinbase', symbol, int(df['timestamp'].max()))
            
            logger.info(f"Coinbase {symbol}: Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse Coinbase OHLC for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_binance_funding(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch funding rate data from Binance"""
        url = REST_BINANCE["funding"].format(symbol=symbol, limit=limit)
        data = await self._fetch_json(url)
        
        if not data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_numeric(df['fundingTime']) // 1000
            df['funding_rate'] = pd.to_numeric(df['fundingRate'])
            df['exchange'] = 'binance'
            df['symbol'] = symbol
            df = df[['timestamp', 'funding_rate', 'exchange', 'symbol']]
            
            logger.info(f"Binance {symbol}: Fetched {len(df)} funding rates")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse Binance funding for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_binance_open_interest(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch open interest data from Binance"""
        url = REST_BINANCE["oi"].format(symbol=symbol, limit=limit)
        data = await self._fetch_json(url)
        
        if not data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_numeric(df['timestamp']) // 1000
            df['open_interest'] = pd.to_numeric(df['sumOpenInterest'])
            df['exchange'] = 'binance'
            df['symbol'] = symbol
            df = df[['timestamp', 'open_interest', 'exchange', 'symbol']]
            
            logger.info(f"Binance {symbol}: Fetched {len(df)} OI records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse Binance OI for {symbol}: {e}")
            return pd.DataFrame()
    
    async def collect_all_data(self, symbols: List[str] = SYMBOLS) -> Dict[str, pd.DataFrame]:
        """Collect data from all exchanges for all symbols"""
        results = {
            'ohlc_binance': [],
            'ohlc_bybit': [],
            'ohlc_coinbase': [],
            'funding_binance': [],
            'oi_binance': []
        }
        
        tasks = []
        for symbol in symbols:
            tasks.append(self.fetch_binance_ohlc(symbol))
            tasks.append(self.fetch_bybit_ohlc(symbol))
            tasks.append(self.fetch_coinbase_ohlc(symbol))
            tasks.append(self.fetch_binance_funding(symbol))
            tasks.append(self.fetch_binance_open_interest(symbol))
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Group results
        for i, symbol in enumerate(symbols):
            idx = i * 5
            if not isinstance(all_results[idx], Exception):
                results['ohlc_binance'].append(all_results[idx])
            if not isinstance(all_results[idx + 1], Exception):
                results['ohlc_bybit'].append(all_results[idx + 1])
            if not isinstance(all_results[idx + 2], Exception):
                results['ohlc_coinbase'].append(all_results[idx + 2])
            if not isinstance(all_results[idx + 3], Exception):
                results['funding_binance'].append(all_results[idx + 3])
            if not isinstance(all_results[idx + 4], Exception):
                results['oi_binance'].append(all_results[idx + 4])
        
        # Concatenate DataFrames
        output = {}
        for key, df_list in results.items():
            if df_list:
                # Filter out empty DataFrames
                non_empty_dfs = [df for df in df_list if not df.empty]
                if non_empty_dfs:
                    combined = pd.concat(non_empty_dfs, ignore_index=True)
                    if not combined.empty:
                        output[key] = combined
                        logger.info(f"{key}: {len(combined)} total rows")
        
        # Ensure we have at least OHLC data
        if not output:
            logger.warning("No data collected from any exchange")
        
        return output


async def test_collector():
    """Test data collector"""
    logger.info("=== Testing Exchange Data Collector ===")
    
    async with ExchangeDataCollector() as collector:
        data = await collector.collect_all_data()
        
        for key, df in data.items():
            logger.info(f"\n{key}:")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Columns: {list(df.columns)}")
            if not df.empty:
                logger.info(f"  Sample:\n{df.head(2)}")
        
        # Validation
        total_rows = sum(len(df) for df in data.values())
        logger.info(f"\n✅ Total rows collected: {total_rows}")
        
        if total_rows > 0:
            logger.info("✅ Data collector test PASSED")
            return True
        else:
            logger.error("❌ Data collector test FAILED - No data collected")
            return False


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        result = asyncio.run(test_collector())
        sys.exit(0 if result else 1)
    else:
        print("Usage: python exchange_data_collector.py --test")

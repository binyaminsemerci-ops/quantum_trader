"""
Exchange Feature Adapter
Create cross-exchange features for AI model training
"""
import redis.asyncio as aioredis
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration - use localhost if not in Docker
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_STREAM_NORMALIZED = "quantum:stream:exchange.normalized"


class ExchangeFeatureAdapter:
    """Extract and transform cross-exchange features"""
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info("✅ Connected to Redis")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def redis_stream_to_df(
        self,
        stream_key: str = REDIS_STREAM_NORMALIZED,
        count: int = 1000
    ) -> pd.DataFrame:
        """Convert Redis stream to Pandas DataFrame"""
        try:
            # Read latest entries from stream
            entries = await self.redis_client.xrevrange(stream_key, count=count)
            
            if not entries:
                logger.warning(f"No data in stream {stream_key}")
                return pd.DataFrame()
            
            # Parse entries into list of dicts
            rows = []
            for entry_id, entry_data in entries:
                # Skip init messages
                if "init" in entry_data:
                    continue
                
                try:
                    row = {
                        "stream_id": entry_id,
                        "symbol": entry_data.get("symbol"),
                        "timestamp": int(entry_data.get("timestamp", 0)),
                        "avg_price": float(entry_data.get("avg_price", 0)),
                        "price_divergence": float(entry_data.get("price_divergence", 0)),
                        "num_exchanges": int(entry_data.get("num_exchanges", 0)),
                        "funding_delta": float(entry_data.get("funding_delta", 0)),
                        "binance_price": self._parse_optional_float(entry_data.get("binance_price")),
                        "bybit_price": self._parse_optional_float(entry_data.get("bybit_price")),
                        "coinbase_price": self._parse_optional_float(entry_data.get("coinbase_price"))
                    }
                    rows.append(row)
                except Exception as e:
                    logger.warning(f"Failed to parse entry {entry_id}: {e}")
            
            df = pd.DataFrame(rows)
            
            if not df.empty:
                # Sort by timestamp
                df = df.sort_values('timestamp')
                logger.info(f"Loaded {len(df)} rows from {stream_key}")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to convert stream to DataFrame: {e}")
            return pd.DataFrame()
    
    def _parse_optional_float(self, value: Optional[str]) -> Optional[float]:
        """Parse optional float value"""
        if value is None or value == "null":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def create_cross_exchange_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from cross-exchange data"""
        if df.empty:
            return df
        
        try:
            # Calculate price momentum (1-minute change)
            df = df.sort_values(['symbol', 'timestamp'])
            df['price_momentum'] = df.groupby('symbol')['avg_price'].pct_change()
            
            # Calculate volatility (rolling std of price divergence)
            df['volatility_spread'] = df.groupby('symbol')['price_divergence'].rolling(
                window=10, min_periods=1
            ).std().reset_index(0, drop=True)
            
            # Exchange price ratios
            df['binance_bybit_ratio'] = np.where(
                (df['binance_price'].notna()) & (df['bybit_price'].notna()) & (df['bybit_price'] != 0),
                df['binance_price'] / df['bybit_price'],
                1.0
            )
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
            
            # Exchange availability indicator
            df['all_exchanges_active'] = (
                df['binance_price'].notna() &
                df['bybit_price'].notna() &
                df['coinbase_price'].notna()
            ).astype(int)
            
            logger.info(f"Created cross-exchange features: {len(df)} rows")
            return df
        
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
            return df
    
    async def get_latest_features(
        self,
        symbols: Optional[List[str]] = None,
        lookback_minutes: int = 60
    ) -> pd.DataFrame:
        """Get latest features for specified symbols"""
        # Get raw data
        df = await self.redis_stream_to_df(count=1000)
        
        if df.empty:
            return df
        
        # Filter by symbols if specified
        if symbols:
            df = df[df['symbol'].isin(symbols)]
        
        # Filter by time window
        cutoff_time = datetime.now().timestamp() - (lookback_minutes * 60)
        df = df[df['timestamp'] >= cutoff_time]
        
        # Create features
        df = self.create_cross_exchange_features(df)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of available feature names"""
        return [
            "avg_price",
            "price_divergence",
            "funding_delta",
            "price_momentum",
            "volatility_spread",
            "binance_bybit_ratio",
            "hour",
            "day_of_week",
            "all_exchanges_active",
            "num_exchanges"
        ]


async def test_feature_adapter():
    """Test feature adapter"""
    logger.info("=== Testing Exchange Feature Adapter ===")
    
    adapter = ExchangeFeatureAdapter()
    await adapter.connect()
    
    try:
        # Get latest features
        df = await adapter.get_latest_features(lookback_minutes=5)
        
        if df.empty:
            logger.warning("No data available - make sure aggregator is running")
            return False
        
        logger.info(f"\n✅ Features DataFrame:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"\nSample data:")
        logger.info(df.head())
        
        logger.info(f"\n✅ Available features:")
        for feature in adapter.get_feature_names():
            logger.info(f"  - {feature}")
        
        logger.info("\n✅ Feature adapter test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ Feature adapter test FAILED: {e}")
        return False
    
    finally:
        await adapter.close()


if __name__ == "__main__":
    import sys
    import asyncio
    
    if "--test" in sys.argv:
        result = asyncio.run(test_feature_adapter())
        sys.exit(0 if result else 1)
    else:
        print("Usage: python exchange_feature_adapter.py --test")

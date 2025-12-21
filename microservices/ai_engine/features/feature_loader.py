"""
Feature Loader for AI Engine
Unified interface for loading features from multiple sources
"""
import pandas as pd
from typing import List, Dict, Optional
import logging
from .exchange_feature_adapter import ExchangeFeatureAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureLoader:
    """Load and merge features from multiple sources"""
    
    def __init__(self):
        self.exchange_adapter = None
    
    async def initialize(self):
        """Initialize feature adapters"""
        self.exchange_adapter = ExchangeFeatureAdapter()
        await self.exchange_adapter.connect()
        logger.info("✅ Feature loader initialized")
    
    async def close(self):
        """Close all connections"""
        if self.exchange_adapter:
            await self.exchange_adapter.close()
    
    async def load_features(
        self,
        feature_source: str = "cross_exchange",
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load features from specified source
        
        Args:
            feature_source: Source of features ('cross_exchange', 'technical', etc.)
            symbols: List of symbols to load
            **kwargs: Additional parameters for specific loaders
        
        Returns:
            DataFrame with features
        """
        if feature_source == "cross_exchange":
            if not self.exchange_adapter:
                await self.initialize()
            
            df = await self.exchange_adapter.get_latest_features(
                symbols=symbols,
                lookback_minutes=kwargs.get('lookback_minutes', 60)
            )
            
            # Get feature list
            features = self.exchange_adapter.get_feature_names()
            logger.info(f"Loaded {len(features)} cross-exchange features")
            
            return df
        
        else:
            logger.warning(f"Unknown feature source: {feature_source}")
            return pd.DataFrame()
    
    def get_available_features(self, feature_source: str = "cross_exchange") -> List[str]:
        """Get list of available features from source"""
        if feature_source == "cross_exchange" and self.exchange_adapter:
            return self.exchange_adapter.get_feature_names()
        return []


async def test_feature_loader():
    """Test feature loader"""
    logger.info("=== Testing Feature Loader ===")
    
    loader = FeatureLoader()
    await loader.initialize()
    
    try:
        # Load cross-exchange features
        df = await loader.load_features(
            feature_source="cross_exchange",
            symbols=["BTCUSDT", "ETHUSDT"],
            lookback_minutes=5
        )
        
        if df.empty:
            logger.warning("No data available - make sure aggregator is running")
            return False
        
        logger.info(f"\n✅ Loaded features:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Features: {loader.get_available_features('cross_exchange')}")
        logger.info(f"\nSample:")
        logger.info(df.head())
        
        logger.info("\n✅ Feature loader test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ Feature loader test FAILED: {e}")
        return False
    
    finally:
        await loader.close()


if __name__ == "__main__":
    import sys
    import asyncio
    
    if "--test" in sys.argv:
        result = asyncio.run(test_feature_loader())
        sys.exit(0 if result else 1)
    else:
        print("Usage: python feature_loader.py --test")

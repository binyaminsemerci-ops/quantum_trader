"""
Binance Client Singleton for Portfolio Service
Provides easy access to configured Binance adapter.
"""
import os
import logging
from typing import Optional
from binance.client import Client

from backend.integrations.exchanges.binance_adapter import BinanceAdapter

logger = logging.getLogger(__name__)

_binance_adapter: Optional[BinanceAdapter] = None


async def get_binance_adapter() -> BinanceAdapter:
    """
    Get or create Binance adapter singleton.
    Reads configuration from environment variables.
    
    Returns:
        BinanceAdapter instance configured for testnet or production
    """
    global _binance_adapter
    
    if _binance_adapter is not None:
        return _binance_adapter
    
    # Read config from env
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
    
    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
    
    # Create Binance client
    client = Client(api_key, api_secret, testnet=use_testnet)
    
    # Set testnet URL if needed
    if use_testnet:
        client.API_URL = 'https://testnet.binancefuture.com/fapi'
        logger.info(f"Binance client configured for TESTNET: {client.API_URL}")
    else:
        logger.info("Binance client configured for PRODUCTION")
    
    # Create adapter
    _binance_adapter = BinanceAdapter(client, testnet=use_testnet)
    
    logger.info(f"BinanceAdapter singleton created (testnet={use_testnet})")
    
    return _binance_adapter


def reset_binance_adapter():
    """Reset singleton (for testing)."""
    global _binance_adapter
    _binance_adapter = None

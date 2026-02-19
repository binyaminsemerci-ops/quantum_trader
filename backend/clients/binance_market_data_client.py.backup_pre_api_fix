"""
Binance Market Data Client for OpportunityRanker
Implements MarketDataClient protocol
"""

import logging
from typing import List
import pandas as pd
import ccxt
from datetime import datetime

logger = logging.getLogger(__name__)


class BinanceMarketDataClient:
    """Real implementation of MarketDataClient using CCXT/Binance."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize Binance client.
        
        Args:
            api_key: Optional Binance API key
            api_secret: Optional Binance API secret
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures market
            }
        })
        logger.info("BinanceMarketDataClient initialized")
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles from Binance.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Candle timeframe (e.g., 1h, 4h, 1d)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            raise
    
    def get_spread(self, symbol: str) -> float:
        """
        Calculate bid-ask spread from orderbook.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Spread percentage (e.g., 0.0005 = 0.05%)
        """
        try:
            # Fetch orderbook
            orderbook = self.exchange.fetch_order_book(symbol, limit=5)
            
            # Get best bid and ask
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            if best_bid > 0 and best_ask > 0:
                spread_pct = (best_ask - best_bid) / best_bid
                return spread_pct
            
            # Default spread if data unavailable
            return 0.001  # 0.1%
            
        except Exception as e:
            logger.warning(f"Failed to get spread for {symbol}: {e}")
            return 0.001  # Default 0.1%
    
    def get_liquidity(self, symbol: str) -> float:
        """
        Get 24h volume in quote currency (USD).
        
        Args:
            symbol: Trading pair
            
        Returns:
            24h volume in USD
        """
        try:
            # Fetch 24h ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Get quote volume (already in USD for USDT pairs)
            volume_usd = ticker.get('quoteVolume', 0)
            
            return float(volume_usd)
            
        except Exception as e:
            logger.warning(f"Failed to get liquidity for {symbol}: {e}")
            return 0.0

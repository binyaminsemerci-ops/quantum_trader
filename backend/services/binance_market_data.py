"""
Binance Futures Market Data Fetcher

Fetches real-time market data for trading profile system:
- 24h volume and price statistics
- Bid/ask spreads
- Orderbook depth
- Funding rates and timing
- Mark price, index price
- Open interest

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException

from backend.services.ai.trading_profile import (
    SymbolMetrics,
    UniverseTier,
    classify_symbol_tier
)

log = logging.getLogger(__name__)


class BinanceMarketDataFetcher:
    """Fetches market data from Binance Futures API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            testnet: Use testnet endpoints
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self._cache: Dict[str, Tuple[SymbolMetrics, datetime]] = {}
        self._cache_ttl = 30  # Cache TTL in seconds
        
    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get 24h ticker statistics.
        
        Returns:
            Dict with volume, price change, high, low, etc.
        """
        try:
            ticker = self.client.futures_ticker(symbol=symbol)
            return ticker
        except BinanceAPIException as e:
            log.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def get_orderbook_depth(
        self,
        symbol: str,
        depth_pct: float = 0.005
    ) -> Tuple[float, float, float]:
        """
        Calculate orderbook depth within ±depth_pct of mid price.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            depth_pct: Percentage range (0.005 = 0.5%)
        
        Returns:
            (bid, ask, depth_notional) where depth is sum of both sides
        """
        try:
            # Get orderbook (limit=100 for reasonable depth)
            book = self.client.futures_order_book(symbol=symbol, limit=100)
            
            bids = book['bids']  # [[price, qty], ...]
            asks = book['asks']
            
            if not bids or not asks:
                return 0.0, 0.0, 0.0
            
            # Best bid/ask
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0
            
            # Calculate depth range
            lower_bound = mid * (1 - depth_pct)
            upper_bound = mid * (1 + depth_pct)
            
            # Sum bid notional within range
            bid_notional = sum(
                float(price) * float(qty)
                for price, qty in bids
                if float(price) >= lower_bound
            )
            
            # Sum ask notional within range
            ask_notional = sum(
                float(price) * float(qty)
                for price, qty in asks
                if float(price) <= upper_bound
            )
            
            total_depth = bid_notional + ask_notional
            
            return best_bid, best_ask, total_depth
            
        except BinanceAPIException as e:
            log.error(f"Failed to get orderbook for {symbol}: {e}")
            return 0.0, 0.0, 0.0
    
    def get_funding_info(self, symbol: str) -> Tuple[float, datetime]:
        """
        Get current funding rate and next funding time.
        
        Returns:
            (funding_rate, next_funding_time)
        """
        try:
            # Get premium index (contains funding rate)
            premium = self.client.futures_funding_rate(symbol=symbol, limit=1)
            
            if not premium:
                return 0.0, datetime.now(timezone.utc)
            
            latest = premium[-1]
            funding_rate = float(latest.get('fundingRate', 0.0))
            funding_time_ms = int(latest.get('fundingTime', 0))
            
            # Next funding time is usually 8h after last funding
            # Binance does funding at 00:00, 08:00, 16:00 UTC
            last_funding = datetime.fromtimestamp(funding_time_ms / 1000, tz=timezone.utc)
            next_funding = last_funding + timedelta(hours=8)
            
            # If next funding is in the past, add 8h
            now = datetime.now(timezone.utc)
            while next_funding < now:
                next_funding += timedelta(hours=8)
            
            return funding_rate, next_funding
            
        except BinanceAPIException as e:
            log.error(f"Failed to get funding info for {symbol}: {e}")
            return 0.0, datetime.now(timezone.utc) + timedelta(hours=8)
    
    def get_mark_price(self, symbol: str) -> Tuple[float, float]:
        """
        Get mark price and index price.
        
        Returns:
            (mark_price, index_price)
        """
        try:
            mark_info = self.client.futures_mark_price(symbol=symbol)
            mark_price = float(mark_info.get('markPrice', 0.0))
            index_price = float(mark_info.get('indexPrice', 0.0))
            return mark_price, index_price
        except BinanceAPIException as e:
            log.error(f"Failed to get mark price for {symbol}: {e}")
            return 0.0, 0.0
    
    def get_open_interest(self, symbol: str) -> float:
        """
        Get current open interest in USDT.
        
        Returns:
            Open interest notional value
        """
        try:
            oi = self.client.futures_open_interest(symbol=symbol)
            # Open interest is in base currency, need to convert to USDT
            oi_qty = float(oi.get('openInterest', 0.0))
            
            # Get mark price to convert
            mark_price, _ = self.get_mark_price(symbol)
            oi_notional = oi_qty * mark_price
            
            return oi_notional
        except BinanceAPIException as e:
            log.error(f"Failed to get open interest for {symbol}: {e}")
            return 0.0
    
    def fetch_symbol_metrics(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[SymbolMetrics]:
        """
        Fetch comprehensive metrics for a single symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            use_cache: Use cached data if available
        
        Returns:
            SymbolMetrics or None if fetch failed
        """
        # Check cache
        if use_cache and symbol in self._cache:
            cached_metrics, cached_time = self._cache[symbol]
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age < self._cache_ttl:
                log.debug(f"Using cached data for {symbol} (age={age:.1f}s)")
                return cached_metrics
        
        try:
            # Fetch all data
            ticker = self.get_24h_ticker(symbol)
            if not ticker:
                return None
            
            bid, ask, depth = self.get_orderbook_depth(symbol)
            funding_rate, next_funding = self.get_funding_info(symbol)
            mark_price, index_price = self.get_mark_price(symbol)
            open_interest = self.get_open_interest(symbol)
            
            # Parse ticker data
            quote_volume = float(ticker.get('quoteVolume', 0.0))
            
            # If bid/ask not set, use ticker prices
            if bid == 0.0 or ask == 0.0:
                bid = float(ticker.get('bidPrice', 0.0))
                ask = float(ticker.get('askPrice', 0.0))
            
            if mark_price == 0.0:
                mark_price = float(ticker.get('lastPrice', 0.0))
            
            # Classify universe tier
            universe_tier = classify_symbol_tier(symbol)
            
            metrics = SymbolMetrics(
                symbol=symbol,
                quote_volume_24h=quote_volume,
                bid=bid,
                ask=ask,
                depth_notional_5bps=depth,
                funding_rate=funding_rate,
                next_funding_time=next_funding,
                mark_price=mark_price,
                index_price=index_price,
                open_interest=open_interest,
                universe_tier=universe_tier
            )
            
            # Cache it
            self._cache[symbol] = (metrics, datetime.now(timezone.utc))
            
            log.debug(
                f"Fetched {symbol}: vol=${quote_volume/1e6:.1f}M, "
                f"spread={((ask-bid)/mark_price)*10000:.2f}bps, "
                f"depth=${depth/1e3:.0f}k, funding={funding_rate*100:.3f}%"
            )
            
            return metrics
            
        except Exception as e:
            log.error(f"Failed to fetch metrics for {symbol}: {e}", exc_info=True)
            return None
    
    def fetch_all_futures_symbols(self) -> List[str]:
        """
        Get list of all USDT perpetual futures symbols.
        
        Returns:
            List of symbol names
        """
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                contract_type = symbol_info.get('contractType', '')
                status = symbol_info.get('status', '')
                
                # Only perpetual futures, only USDT quote, only trading
                if (contract_type == 'PERPETUAL' and
                    symbol.endswith('USDT') and
                    status == 'TRADING'):
                    symbols.append(symbol)
            
            log.info(f"Found {len(symbols)} USDT perpetual futures")
            return symbols
            
        except BinanceAPIException as e:
            log.error(f"Failed to get futures symbols: {e}")
            return []
    
    def fetch_universe_metrics(
        self,
        symbols: Optional[List[str]] = None,
        max_symbols: int = 100
    ) -> List[SymbolMetrics]:
        """
        Fetch metrics for multiple symbols (or all available).
        
        Args:
            symbols: List of symbols to fetch, or None for all
            max_symbols: Maximum symbols to fetch
        
        Returns:
            List of SymbolMetrics
        """
        if symbols is None:
            symbols = self.fetch_all_futures_symbols()
        
        # Limit to avoid rate limits
        symbols = symbols[:max_symbols]
        
        metrics_list = []
        
        for symbol in symbols:
            metrics = self.fetch_symbol_metrics(symbol)
            if metrics:
                metrics_list.append(metrics)
            
            # Small delay to avoid rate limits (2400 req/min = ~40 req/s)
            # Safe to do 10 req/s
            asyncio.sleep(0.1)
        
        log.info(f"Fetched metrics for {len(metrics_list)}/{len(symbols)} symbols")
        return metrics_list
    
    def clear_cache(self):
        """Clear the metrics cache."""
        self._cache.clear()
        log.debug("Metrics cache cleared")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_market_data_fetcher(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False
) -> BinanceMarketDataFetcher:
    """
    Factory function to create market data fetcher.
    
    Reads API credentials from environment if not provided.
    """
    import os
    
    if api_key is None:
        api_key = os.getenv('BINANCE_API_KEY')
    
    if api_secret is None:
        api_secret = os.getenv('BINANCE_API_SECRET')
    
    # Check GO-LIVE status: if go_live.active exists, use production
    from pathlib import Path
    go_live_active = Path("go_live.active").exists()
    testnet_env = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    testnet = (testnet or testnet_env) and not go_live_active
    
    return BinanceMarketDataFetcher(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )


# ============================================================================
# ATR CALCULATION
# ============================================================================

def calculate_atr(
    symbol: str,
    period: int = 14,
    timeframe: str = '15m',
    client: Optional[Client] = None
) -> Optional[float]:
    """
    Calculate Average True Range for a symbol.
    
    Args:
        symbol: Trading pair
        period: ATR period (default 14)
        timeframe: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
        client: Binance client (creates new if None)
    
    Returns:
        ATR value or None if calculation failed
    """
    if client is None:
        import os
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        client = Client(api_key, api_secret)
    
    try:
        # Fetch klines (need period + 1 for ATR calculation)
        klines = client.futures_klines(
            symbol=symbol,
            interval=timeframe,
            limit=period + 50  # Get extra for stability
        )
        
        if len(klines) < period + 1:
            log.error(f"Not enough klines for ATR calculation: {len(klines)}")
            return None
        
        # Calculate True Range for each candle
        true_ranges = []
        
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i-1][4])
            
            # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Calculate ATR as simple moving average of True Range
        if len(true_ranges) < period:
            log.error(f"Not enough TR values: {len(true_ranges)}")
            return None
        
        atr = sum(true_ranges[-period:]) / period
        
        log.debug(f"ATR for {symbol} ({timeframe}, {period}): {atr:.8f}")
        return atr
        
    except Exception as e:
        log.error(f"Failed to calculate ATR for {symbol}: {e}", exc_info=True)
        return None


def calculate_atr_percentage(
    symbol: str,
    period: int = 14,
    timeframe: str = '15m',
    client: Optional[Client] = None
) -> Optional[float]:
    """
    Calculate ATR as percentage of current price.
    
    Returns:
        ATR/Price ratio (e.g., 0.02 = 2%)
    """
    atr = calculate_atr(symbol, period, timeframe, client)
    if atr is None:
        return None
    
    # Get current price
    if client is None:
        import os
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        client = Client(api_key, api_secret)
    
    try:
        ticker = client.futures_ticker(symbol=symbol)
        price = float(ticker.get('lastPrice', 0.0))
        
        if price == 0:
            return None
        
        atr_pct = atr / price
        return atr_pct
        
    except Exception as e:
        log.error(f"Failed to get price for {symbol}: {e}")
        return None

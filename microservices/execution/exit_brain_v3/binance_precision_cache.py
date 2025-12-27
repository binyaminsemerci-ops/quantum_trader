"""
Binance Exchange Info Cache for real precision values.

Fetches actual tick_size and step_size from Binance API instead of guessing.
"""
import logging
from typing import Dict, Tuple, Optional
import time

logger = logging.getLogger(__name__)

# Global cache for exchange info
_EXCHANGE_INFO_CACHE: Dict[str, Tuple[float, float]] = {}
_CACHE_TIMESTAMP: float = 0
_CACHE_TTL_SECONDS = 3600  # 1 hour


def get_binance_precision_from_api(symbol: str, client) -> Tuple[float, float]:
    """
    Get real tick_size and step_size from Binance API.
    
    Args:
        symbol: Trading pair (e.g., "XRPUSDT")
        client: Binance client instance
        
    Returns:
        (tick_size, step_size) tuple with real values from exchange
    """
    global _EXCHANGE_INFO_CACHE, _CACHE_TIMESTAMP
    
    # Check cache
    current_time = time.time()
    if symbol in _EXCHANGE_INFO_CACHE and (current_time - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS:
        return _EXCHANGE_INFO_CACHE[symbol]
    
    # Refresh cache if stale
    if (current_time - _CACHE_TIMESTAMP) >= _CACHE_TTL_SECONDS:
        try:
            logger.info("[PRECISION_CACHE] Refreshing exchange info from Binance API...")
            exchange_info = client.futures_exchange_info()
            
            for symbol_info in exchange_info.get('symbols', []):
                sym = symbol_info['symbol']
                
                # Extract PRICE filter (tick_size)
                price_filter = next(
                    (f for f in symbol_info.get('filters', []) if f['filterType'] == 'PRICE_FILTER'),
                    None
                )
                tick_size = float(price_filter['tickSize']) if price_filter else 0.001
                
                # Extract LOT_SIZE filter (step_size)
                lot_filter = next(
                    (f for f in symbol_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'),
                    None
                )
                step_size = float(lot_filter['stepSize']) if lot_filter else 1.0
                
                _EXCHANGE_INFO_CACHE[sym] = (tick_size, step_size)
            
            _CACHE_TIMESTAMP = current_time
            logger.info(
                f"[PRECISION_CACHE] Cached precision for {len(_EXCHANGE_INFO_CACHE)} symbols"
            )
            
        except Exception as e:
            logger.error(f"[PRECISION_CACHE] Failed to fetch exchange info: {e}")
            # Fall back to hardcoded defaults if API fails
            return _get_default_precision(symbol)
    
    # Return from cache
    if symbol in _EXCHANGE_INFO_CACHE:
        tick, step = _EXCHANGE_INFO_CACHE[symbol]
        logger.debug(f"[PRECISION_CACHE] {symbol}: tick_size={tick}, step_size={step} (from API)")
        return tick, step
    
    # Fallback to defaults if symbol not found
    logger.warning(f"[PRECISION_CACHE] {symbol} not found in exchange info, using defaults")
    return _get_default_precision(symbol)


def _get_default_precision(symbol: str) -> Tuple[float, float]:
    """Fallback precision if API unavailable."""
    if 'BTC' in symbol or 'ETH' in symbol:
        return (0.1, 0.001)
    elif any(coin in symbol for coin in ['TRX', 'DOGE', 'XRP', 'SHIB', 'WIN']):
        return (0.00001, 1.0)
    elif 'DOT' in symbol or 'LINK' in symbol or 'UNI' in symbol:
        return (0.001, 0.1)
    else:
        return (0.001, 1.0)


def clear_cache():
    """Clear precision cache (useful for testing)."""
    global _EXCHANGE_INFO_CACHE, _CACHE_TIMESTAMP
    _EXCHANGE_INFO_CACHE.clear()
    _CACHE_TIMESTAMP = 0
    logger.info("[PRECISION_CACHE] Cache cleared")

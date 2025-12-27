"""
Precision helpers for Binance order submission.

Ensures prices and quantities conform to exchange tickSize/stepSize rules.
"""
from decimal import Decimal, ROUND_DOWN
import logging
from typing import Dict, Tuple, Optional, Any
import time

logger = logging.getLogger(__name__)

# Global cache for exchange info filters
_EXCHANGE_INFO_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TIMESTAMP: float = 0
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _to_decimal(x) -> Decimal:
    """Convert value to Decimal for precise calculations."""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def quantize_to_tick(price: float, tick_size: float) -> float:
    """
    Snap price down to nearest valid tick.
    
    Args:
        price: Raw price value
        tick_size: Minimum price increment for symbol (from exchangeInfo)
        
    Returns:
        Price rounded down to nearest tick
        
    Example:
        price=1.234567, tick_size=0.0001 -> 1.2345
        price=0.27994, tick_size=0.00001 -> 0.27994
        price=875.41138159, tick_size=0.001 -> 875.411
    """
    p = _to_decimal(price)
    t = _to_decimal(tick_size)
    
    # Use quantize with ROUND_DOWN to match tick precision
    quantized = p.quantize(t, rounding=ROUND_DOWN)
    
    # Format to string with proper precision, then back to float to avoid trailing zeros
    # Determine decimal places from tick_size (handle scientific notation)
    tick_decimal = _to_decimal(tick_size)
    # Get number of decimal places by converting to tuple
    # tuple()[2] gives the negative exponent (decimal places)
    decimals = abs(tick_decimal.as_tuple().exponent)
    
    result = float(f"{quantized:.{decimals}f}")
    
    if result != float(price):
        logger.debug(
            f"[PRECISION] Price adjusted: {price:.8f} → {result:.{decimals}f} (tick={tick_size})"
        )
    
    return result


def quantize_to_step(qty: float, step_size: float) -> float:
    """
    Snap quantity down to nearest valid step.
    
    Args:
        qty: Raw quantity value
        step_size: Minimum quantity increment for symbol (from exchangeInfo)
        
    Returns:
        Quantity rounded down to nearest step
        
    Example:
        qty=144699.123, step_size=1.0 -> 144699.0
        qty=4.267891, step_size=0.001 -> 4.267
    """
    q = _to_decimal(qty)
    s = _to_decimal(step_size)
    
    # Use quantize with ROUND_DOWN to match step precision
    quantized = q.quantize(s, rounding=ROUND_DOWN)
    
    # Format to string with proper precision, then back to float
    step_str = str(step_size).rstrip('0').rstrip('.')
    if '.' in step_str:
        decimals = len(step_str.split('.')[1])
    else:
        decimals = 0
    
    result = float(f"{quantized:.{decimals}f}")
    
    if result != float(qty):
        logger.debug(
            f"[PRECISION] Quantity adjusted: {qty:.8f} → {result:.{decimals}f} (step={step_size})"
        )
    
    return result


def get_binance_precision(symbol: str) -> tuple[float, float]:
    """
    Get tickSize and stepSize for a symbol.
    
    Common Binance Futures precision patterns:
    - BTC/ETH: tick=0.1, step=0.001
    - Most alts: tick=0.001, step=1.0 or step=0.1
    - Low price coins (TRX/DOGE/XRP): tick=0.00001, step=1.0
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        
    Returns:
        (tick_size, step_size) tuple
        
    Note:
        This uses common defaults. For production, cache exchangeInfo 
        and look up actual values per symbol.
    """
    # Default precision based on symbol patterns
    if 'BTC' in symbol or 'ETH' in symbol:
        tick_size = 0.1
        step_size = 0.001
    elif any(coin in symbol for coin in ['TRX', 'DOGE', 'XRP', 'SHIB', 'WIN']):
        # Very low price coins
        tick_size = 0.00001
        step_size = 1.0
    elif 'DOT' in symbol or 'LINK' in symbol or 'UNI' in symbol:
        # Mid-range coins
        tick_size = 0.001
        step_size = 0.1
    else:
        # Default for most altcoins
        tick_size = 0.001
        step_size = 1.0
    
    logger.debug(
        f"[PRECISION] {symbol}: tick_size={tick_size}, step_size={step_size}"
    )
    
    return tick_size, step_size


def format_price_for_binance(price: float, tick_size: float) -> str:
    """
    Format price as string with exact decimal places required by Binance.
    
    Binance requires stopPrice/price to have correct number of decimals.
    Example: tick_size=0.00001 requires 5 decimals ("1.95830" not "1.9583")
    
    Args:
        price: Price value (already quantized to tick_size)
        tick_size: Minimum price increment
        
    Returns:
        Price formatted as string with correct decimal places
        
    Example:
        price=1.9583, tick_size=0.00001 -> "1.95830"
        price=0.275, tick_size=0.001 -> "0.275"
    """
    # Determine decimal places from tick_size
    tick_decimal = _to_decimal(tick_size)
    decimals = abs(tick_decimal.as_tuple().exponent)
    
    # Format price with exact decimals
    return f"{price:.{decimals}f}"


def get_symbol_filters(symbol: str, client) -> Dict[str, Any]:
    """
    Get exchange filters for a symbol from Binance exchangeInfo.
    
    Caches results in memory for 1 hour to avoid repeated API calls.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        client: Binance client with futures_exchange_info method
        
    Returns:
        Dict with keys: tick_size, step_size, min_qty, max_qty, min_notional
        
    Raises:
        ValueError: If symbol not found in exchange info
    """
    global _EXCHANGE_INFO_CACHE, _CACHE_TIMESTAMP
    
    current_time = time.time()
    
    # Check cache validity
    if symbol in _EXCHANGE_INFO_CACHE and (current_time - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS:
        return _EXCHANGE_INFO_CACHE[symbol]
    
    # Refresh cache if stale
    if (current_time - _CACHE_TIMESTAMP) >= _CACHE_TTL_SECONDS:
        try:
            logger.debug("[PRECISION] Refreshing exchange info cache from Binance API")
            exchange_info = client.futures_exchange_info()
            
            for symbol_info in exchange_info.get('symbols', []):
                sym = symbol_info['symbol']
                filters = {}
                
                # Extract PRICE_FILTER
                for f in symbol_info.get('filters', []):
                    if f['filterType'] == 'PRICE_FILTER':
                        filters['tick_size'] = float(f['tickSize'])
                        filters['min_price'] = float(f.get('minPrice', 0))
                        filters['max_price'] = float(f.get('maxPrice', 1000000000))
                    elif f['filterType'] == 'LOT_SIZE':
                        filters['step_size'] = float(f['stepSize'])
                        filters['min_qty'] = float(f['minQty'])
                        filters['max_qty'] = float(f.get('maxQty', 1000000000))
                    elif f['filterType'] == 'MIN_NOTIONAL':
                        filters['min_notional'] = float(f.get('notional', 0))
                
                # Set defaults if filters missing
                filters.setdefault('tick_size', 0.001)
                filters.setdefault('step_size', 1.0)
                filters.setdefault('min_qty', 0.001)
                filters.setdefault('max_qty', 1000000000)
                filters.setdefault('min_notional', 5.0)
                
                _EXCHANGE_INFO_CACHE[sym] = filters
            
            _CACHE_TIMESTAMP = current_time
            logger.info(
                f"[PRECISION] Cached filters for {len(_EXCHANGE_INFO_CACHE)} symbols"
            )
            
        except Exception as e:
            logger.error(f"[PRECISION] Failed to fetch exchange info: {e}")
            # Return defaults for this symbol if cache refresh failed
            if symbol not in _EXCHANGE_INFO_CACHE:
                return _get_default_filters(symbol)
    
    # Return from cache
    if symbol in _EXCHANGE_INFO_CACHE:
        return _EXCHANGE_INFO_CACHE[symbol]
    
    # Fallback to defaults
    logger.warning(f"[PRECISION] Symbol {symbol} not found in exchange info, using defaults")
    return _get_default_filters(symbol)


def _get_default_filters(symbol: str) -> Dict[str, Any]:
    """Fallback filters based on common patterns."""
    if 'BTC' in symbol or 'ETH' in symbol:
        return {
            'tick_size': 0.1,
            'step_size': 0.001,
            'min_qty': 0.001,
            'max_qty': 1000000000,
            'min_notional': 5.0
        }
    elif any(coin in symbol for coin in ['TRX', 'DOGE', 'XRP', 'SHIB', 'WIN']):
        return {
            'tick_size': 0.00001,
            'step_size': 1.0,
            'min_qty': 1.0,
            'max_qty': 1000000000,
            'min_notional': 5.0
        }
    else:
        return {
            'tick_size': 0.001,
            'step_size': 1.0,
            'min_qty': 1.0,
            'max_qty': 1000000000,
            'min_notional': 5.0
        }


def quantize_price(symbol: str, price: float, client) -> float:
    """
    Quantize price to comply with symbol's tick_size.
    
    Rounds DOWN to avoid 'precision over maximum' errors.
    
    Args:
        symbol: Trading pair
        price: Raw price value
        client: Binance client for fetching filters
        
    Returns:
        Price rounded down to nearest valid tick
    """
    filters = get_symbol_filters(symbol, client)
    tick_size = filters['tick_size']
    return quantize_to_tick(price, tick_size)


def quantize_stop_price(symbol: str, stop_price: float, client) -> float:
    """
    Quantize stopPrice to comply with symbol's tick_size.
    
    Same as quantize_price but kept separate for clarity.
    
    Args:
        symbol: Trading pair
        stop_price: Raw stop price value
        client: Binance client for fetching filters
        
    Returns:
        Stop price rounded down to nearest valid tick
    """
    filters = get_symbol_filters(symbol, client)
    tick_size = filters['tick_size']
    return quantize_to_tick(stop_price, tick_size)


def quantize_quantity(symbol: str, quantity: float, client) -> float:
    """
    Quantize quantity to comply with symbol's LOT_SIZE.
    
    Rounds DOWN and ensures result >= min_qty.
    
    Args:
        symbol: Trading pair
        quantity: Raw quantity value
        client: Binance client for fetching filters
        
    Returns:
        Quantity rounded down to nearest valid step, or 0.0 if below min_qty
    """
    filters = get_symbol_filters(symbol, client)
    step_size = filters['step_size']
    min_qty = filters['min_qty']
    
    quantized = quantize_to_step(quantity, step_size)
    
    if quantized < min_qty:
        logger.warning(
            f"[PRECISION] {symbol}: Quantity {quantized} below min_qty {min_qty}, returning 0.0"
        )
        return 0.0
    
    return quantized


def clear_precision_cache():
    """Clear the exchange info cache (useful for testing)."""
    global _EXCHANGE_INFO_CACHE, _CACHE_TIMESTAMP
    _EXCHANGE_INFO_CACHE.clear()
    _CACHE_TIMESTAMP = 0
    logger.info("[PRECISION] Cache cleared")

"""
Funding Rate Filter - Automatic blacklisting of high-funding-fee symbols

Fetches current funding rates from Binance and filters out symbols with
extreme funding fees that make them unprofitable regardless of exit timing.

Normal funding: ~0.01% per 8h (0.03% per day)
Extreme funding: >0.1% per 8h (0.3% per day) → blacklisted

Examples of problematic symbols:
- BREVUSDT: 78 USDT funding on 31 USDT margin (250% ratio)
- API3USDT: 150 USDT funding on 158 USDT margin (95% ratio)
"""
import asyncio
import logging
from typing import Dict, List, Set
import aiohttp

logger = logging.getLogger(__name__)


# Thresholds for automatic blacklisting
EXTREME_FUNDING_RATE_THRESHOLD = 0.001  # 0.1% per 8h (10× normal)
NORMAL_FUNDING_RATE = 0.0001  # 0.01% per 8h (typical)

# Manual permanent blacklist (known problematic symbols)
MANUAL_BLACKLIST = {
    "BREVUSDT",   # 78 USDT funding on 31 USDT margin
    "API3USDT",   # 150 USDT funding on 158 USDT margin
}


async def fetch_funding_rate(symbol: str) -> float:
    """
    Fetch current funding rate from Binance Futures API
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
    
    Returns:
        Funding rate as float (e.g., 0.0001 = 0.01%)
        Returns 0.0 if unavailable or API error
    
    API: https://fapi.binance.com/fapi/v1/fundingRate
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": 1}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        rate = float(data[0].get("fundingRate", 0.0))
                        return rate
                else:
                    logger.debug(f"Funding rate API error for {symbol}: HTTP {response.status}")
                    return 0.0
    except asyncio.TimeoutError:
        logger.debug(f"Funding rate fetch timeout for {symbol}")
        return 0.0
    except Exception as e:
        logger.debug(f"Funding rate fetch error for {symbol}: {e}")
        return 0.0


async def filter_symbols_by_funding_rate(
    symbols: List[str],
    threshold: float = EXTREME_FUNDING_RATE_THRESHOLD
) -> Dict[str, any]:
    """
    Filter symbols by funding rate, blacklisting those with extreme rates
    
    Args:
        symbols: List of symbol names to check
        threshold: Funding rate threshold (default 0.001 = 0.1% per 8h)
    
    Returns:
        Dict with:
        - "allowed": List of symbols with acceptable funding rates
        - "blacklisted": List of symbols with extreme funding rates
        - "rates": Dict of {symbol: funding_rate}
        - "reasons": Dict of {symbol: reason_string}
    """
    logger.info(f"[FundingFilter] Checking funding rates for {len(symbols)} symbols...")
    
    # Fetch rates concurrently
    tasks = [fetch_funding_rate(symbol) for symbol in symbols]
    rates = await asyncio.gather(*tasks)
    
    # Build rate map
    rate_map = dict(zip(symbols, rates))
    
    allowed = []
    blacklisted = []
    reasons = {}
    
    for symbol, rate in rate_map.items():
        # Check manual blacklist first
        if symbol in MANUAL_BLACKLIST:
            blacklisted.append(symbol)
            reasons[symbol] = f"MANUAL_BLACKLIST (known problematic)"
            logger.warning(f"[FundingFilter] ❌ {symbol}: MANUAL_BLACKLIST")
            continue
        
        # Check if rate is extreme
        abs_rate = abs(rate)
        
        if abs_rate > threshold:
            blacklisted.append(symbol)
            rate_pct = abs_rate * 100
            daily_rate_pct = rate_pct * 3  # 3 cycles per day
            reasons[symbol] = f"Extreme funding: {rate_pct:.4f}% per 8h ({daily_rate_pct:.4f}% per day)"
            logger.warning(f"[FundingFilter] ❌ {symbol}: {reasons[symbol]}")
        else:
            allowed.append(symbol)
            if rate != 0.0:
                logger.debug(f"[FundingFilter] ✅ {symbol}: {rate*100:.4f}% per 8h (OK)")
    
    logger.info(f"[FundingFilter] Results: {len(allowed)} allowed, {len(blacklisted)} blacklisted")
    
    if blacklisted:
        logger.warning(f"[FundingFilter] Blacklisted symbols: {', '.join(blacklisted)}")
    
    return {
        "allowed": allowed,
        "blacklisted": blacklisted,
        "rates": rate_map,
        "reasons": reasons
    }


async def get_filtered_symbols(
    candidate_symbols: List[str],
    enable_funding_filter: bool = True,
    threshold: float = EXTREME_FUNDING_RATE_THRESHOLD
) -> List[str]:
    """
    Get list of symbols safe to trade (funding-rate filtered)
    
    Args:
        candidate_symbols: Initial list of symbols to consider
        enable_funding_filter: Enable automatic funding rate filtering
        threshold: Funding rate threshold
    
    Returns:
        List of symbols passing funding rate checks
    
    Usage:
        symbols = ["BTCUSDT", "ETHUSDT", "BREVUSDT", "API3USDT"]
        safe_symbols = await get_filtered_symbols(symbols)
        # Returns: ["BTCUSDT", "ETHUSDT"]
    """
    if not enable_funding_filter:
        logger.info("[FundingFilter] Funding filter DISABLED - using all symbols")
        return candidate_symbols
    
    result = await filter_symbols_by_funding_rate(candidate_symbols, threshold)
    
    allowed = result["allowed"]
    blacklisted = result["blacklisted"]
    
    if blacklisted:
        logger.info(f"[FundingFilter] 🛡️ Protected from {len(blacklisted)} high-fee symbols")
        for symbol in blacklisted:
            reason = result["reasons"].get(symbol, "Unknown")
            logger.info(f"  ❌ {symbol}: {reason}")
    
    return allowed


async def refresh_funding_filter(
    entry_scanner,
    threshold: float = EXTREME_FUNDING_RATE_THRESHOLD
):
    """
    Refresh funding rate filter for an existing EntryScanner
    
    Updates the scanner's symbol list by re-filtering with current rates.
    Useful for periodic refresh to catch rate changes.
    
    Args:
        entry_scanner: EntryScanner instance to update
        threshold: Funding rate threshold
    """
    original_count = len(entry_scanner.symbols)
    
    logger.info(f"[FundingFilter] Refreshing filter for {original_count} symbols...")
    
    result = await filter_symbols_by_funding_rate(entry_scanner.symbols, threshold)
    
    new_symbols = result["allowed"]
    newly_blacklisted = result["blacklisted"]
    
    entry_scanner.symbols = new_symbols
    
    logger.info(f"[FundingFilter] Refresh complete: {len(new_symbols)} allowed (removed {len(newly_blacklisted)})")
    
    if newly_blacklisted:
        logger.warning(f"[FundingFilter] Newly blacklisted: {', '.join(newly_blacklisted)}")

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json

logger = logging.getLogger(__name__)

# Binance Testnet USDT Perpetual Futures - Verified symbols (Nov 2025)
# These are the ONLY symbols available on testnet.binancefuture.com
TESTNET_WHITELIST = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "DOTUSDT", "TRXUSDT",
    "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "LTCUSDT",
    "APTUSDT", "SUIUSDT", "SEIUSDT", "OPUSDT", "ARBUSDT", "TONUSDT"
]

# Known invalid/delisted symbols that cause timeouts
# These symbols are NOT available on Binance Futures but may appear in API results
INVALID_SYMBOLS_BLACKLIST = [
    "PENGUUSDT", "LIGHTUSDT", "TAOUSDT", "USTCUSDT", "FOLKSUSDT",
    "LUNAUSTDT",  # Deprecated after Terra collapse
    # Add more invalid symbols here as they are discovered
]


def _fetch_binance_24h() -> List[Dict]:
    """Fetch 24h ticker stats from Binance FUTURES API (USDT perpetual contracts).

    Returns a list of dicts with fields including 'symbol' and 'quoteVolume'.
    Uses requests to avoid adding async complexity here.
    """
    import requests  # type: ignore

    # Use FUTURES API endpoint for perpetual contracts
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return []


def _volume_map_for_quote(quote: str) -> Dict[str, float]:
    """Build a map symbol->quoteVolume for FUTURES contracts (USDC/USDT perpetual)."""
    # Use the requested quote (USDC or USDT)
    quote = quote.upper()
    
    items = _fetch_binance_24h()
    out: Dict[str, float] = {}
    for it in items:
        try:
            sym = str(it.get("symbol", ""))
            if not sym.endswith(quote):
                continue
            # Skip blacklisted invalid symbols
            if sym in INVALID_SYMBOLS_BLACKLIST:
                logger.debug(f"Skipping blacklisted symbol: {sym}")
                continue
            vol = float(it.get("quoteVolume", 0.0))
            out[sym] = vol
        except Exception:
            continue
    return out


def _candidate_coins() -> List[str]:
    """Return base + L1 + L2 coin tickers without quote suffix.

    Pulls from config.trading_universe, falling back to common majors.
    """
    try:
        from config.trading_universe import LAYER1_COINS, LAYER2_COINS, DEFAULT_UNIVERSE  # type: ignore
        # Strip quote suffix if present in DEFAULT_UNIVERSE to infer base coins
        base_from_default: List[str] = []
        for s in DEFAULT_UNIVERSE:
            # Symbols like BTCUSDT -> coin = BTC
            if len(s) > 4:
                base_from_default.append(s[:-4])
        # ensure uniqueness while keeping order
        seen = set()
        base = [c for c in base_from_default if not (c in seen or seen.add(c))]
        # Build combined
        layer1 = [c[:-4] if len(c) > 4 else c for c in LAYER1_COINS]
        layer2 = [c[:-4] if len(c) > 4 else c for c in LAYER2_COINS]
        coins = base + layer1 + layer2
        # Deduplicate preserve order
        seen2 = set()
        return [c for c in coins if c and not (c in seen2 or seen2.add(c))]
    except Exception:
        # Fallback hardcoded majors if imports fail
        return [
            "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "LINK", "AVAX",
            "DOT", "MATIC", "ATOM", "ARB", "OP", "UNI", "SHIB", "APT", "NEAR", "LTC",
        ]


def get_l1l2_top_by_volume(quote: str = "USDC", max_symbols: int = 100) -> List[str]:
    """Return L1/L2 + base coins prioritized by highest 24h quote volume (FUTURES ONLY).

    Uses CoinGecko for top coins + Binance Futures for validation.
    Combines CoinGecko market data with Binance futures availability.
    """
    # Use requested quote (default USDC)
    quote = quote.upper()
    
    # Try to use CoinGecko-based universe manager for better data
    try:
        import asyncio
        from backend.services.coingecko.top_coins_fetcher import TopCoinsFetcher
        
        # Fetch from CoinGecko
        fetcher = TopCoinsFetcher()
        loop = asyncio.get_event_loop()
        
        # Run async fetch
        if loop.is_running():
            # If event loop is already running, use cached data
            symbols = fetcher._load_cache()
            if symbols:
                logger.info(f"[UNIVERSE] Using cached CoinGecko data: {len(symbols)} coins")
                # Convert to Binance symbols
                binance_symbols = [coin["binance_symbol"] for coin in symbols[:max_symbols]]
                
                # Validate against Binance Futures
                volume_map = _volume_map_for_quote(quote)
                valid_symbols = [s for s in binance_symbols if s in volume_map]
                
                if len(valid_symbols) >= max_symbols * 0.7:  # At least 70% valid
                    logger.info(f"[UNIVERSE] CoinGecko universe: {len(valid_symbols)} valid symbols")
                    return valid_symbols[:max_symbols]
                else:
                    logger.warning(f"[UNIVERSE] CoinGecko validation low ({len(valid_symbols)}), falling back")
        else:
            # Event loop not running, can use asyncio.run
            coins = asyncio.run(fetcher.fetch_top_coins(max_symbols, force_refresh=False))
            if coins and len(coins) > 0:
                logger.info(f"[UNIVERSE] Fetched {len(coins)} coins from CoinGecko")
                binance_symbols = [coin["binance_symbol"] for coin in coins]
                
                # Validate against Binance Futures
                volume_map = _volume_map_for_quote(quote)
                valid_symbols = [s for s in binance_symbols if s in volume_map]
                
                if len(valid_symbols) >= max_symbols * 0.7:
                    logger.info(f"[UNIVERSE] CoinGecko universe: {len(valid_symbols)} valid symbols")
                    return valid_symbols[:max_symbols]
    except Exception as e:
        logger.warning(f"[UNIVERSE] CoinGecko fetch failed: {e}, using Binance fallback")
    
    # Fallback: Use original Binance-only logic
    coins = _candidate_coins()

    # Get futures volume map for requested quote
    volume_map = _volume_map_for_quote(quote)

    ranked: List[Tuple[str, float]] = []  # (symbol, volume)
    for coin in coins:
        sym = f"{coin}{quote}"  # Use dynamic quote instead of hardcoded USDT
        if sym in volume_map:
            ranked.append((sym, volume_map[sym]))

    # Sort by volume desc
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"[UNIVERSE] Binance fallback: {len(ranked)} symbols ranked")

    # Return top symbols
    return [sym for sym, _vol in ranked[:max_symbols]]


def get_megacap_universe(quote: str = "USDT", max_symbols: int = 50) -> List[str]:
    """Return top megacap symbols (highest volume majors only).
    
    This is a SAFE default for production environments.
    """
    # Hardcoded major cryptocurrencies
    megacap_coins = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX",
        "LINK", "AVAX", "DOT", "MATIC", "UNI", "LTC", "ATOM",
        "NEAR", "APT", "ARB", "OP", "SUI", "SHIB", "TON"
    ]
    
    quote = quote.upper()
    volume_map = _volume_map_for_quote(quote)
    
    ranked: List[Tuple[str, float]] = []
    for coin in megacap_coins:
        sym = f"{coin}{quote}"
        if sym in volume_map:
            ranked.append((sym, volume_map[sym]))
    
    # Sort by volume desc
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return [sym for sym, _vol in ranked[:max_symbols]]


def get_all_usdt_universe(quote: str = "USDT", max_symbols: int = 300) -> List[str]:
    """Return ALL available USDT perpetual futures, sorted by volume.
    
    This fetches ALL symbols from Binance Futures and ranks them by 24h volume.
    """
    quote = quote.upper()
    volume_map = _volume_map_for_quote(quote)
    
    # Sort all symbols by volume
    ranked = sorted(volume_map.items(), key=lambda x: x[1], reverse=True)
    
    return [sym for sym, _vol in ranked[:max_symbols]]


def get_l1l2_top_by_volume_multi(quotes: List[str], max_symbols: int = 100) -> List[str]:
    """Return L1/L2 + base coin pairs for FUTURES (cross-margin multi-quote support).

    - Supports both USDC and USDT futures for cross-margin trading
    - Ranks coins by 24h futures volume across all requested quotes
    - Returns up to `max_symbols` perpetual contracts
    """
    if not quotes:
        quotes = ["USDC", "USDT"]  # Default to both for cross-margin
    
    quotes = [q.upper() for q in quotes]
    coins = _candidate_coins()
    
    # Get volume maps for all quotes
    volume_maps = {q: _volume_map_for_quote(q) for q in quotes}
    
    # Rank coins by max volume across all quotes
    coin_rankings: List[Tuple[str, float, str]] = []  # (coin, max_volume, best_quote)
    for coin in coins:
        max_vol = 0.0
        best_quote = quotes[0]
        for quote in quotes:
            sym = f"{coin}{quote}"
            vol = volume_maps.get(quote, {}).get(sym, 0.0)
            if vol > max_vol:
                max_vol = vol
                best_quote = quote
        if max_vol > 0.0:
            coin_rankings.append((coin, max_vol, best_quote))
    
    # Sort by volume
    coin_rankings.sort(key=lambda x: x[1], reverse=True)
    
    # Generate symbols with both USDC and USDT for cross-margin
    symbols: List[str] = []
    for coin, _vol, primary_quote in coin_rankings[:max_symbols]:
        # Add primary quote symbol
        symbols.append(f"{coin}{primary_quote}")
        
        # Add secondary quote if available and within limit
        if len(symbols) < max_symbols:
            for quote in quotes:
                if quote != primary_quote:
                    sym = f"{coin}{quote}"
                    if sym in volume_maps.get(quote, {}):
                        symbols.append(sym)
                        break
        
        if len(symbols) >= max_symbols:
            break
    
    return symbols[:max_symbols]


# ==============================================================================
# MAIN UNIVERSE LOADER
# ==============================================================================

def load_universe(
    universe_name: str,
    max_symbols: int = 300,
    quote: str = "USDT",
    exchange: Optional[Any] = None
) -> List[str]:
    """Load trading universe based on profile name.
    
    Args:
        universe_name: Profile name ("l1l2-top", "megacap", "all-usdt")
        max_symbols: Maximum number of symbols to return
        quote: Quote currency (USDT or USDC)
        exchange: Optional exchange adapter (currently unused, for future expansion)
    
    Returns:
        List of symbol strings (e.g. ["BTCUSDT", "ETHUSDT", ...])
    
    Supported universe profiles:
        - "l1l2-top": Layer 1 + Layer 2 + major coins, sorted by volume
        - "megacap": High marketcap majors only (SAFE default)
        - "all-usdt": All available USDT perpetual futures
    """
    universe_name = universe_name.lower().strip()
    quote = quote.upper()
    
    # Check if running on testnet
    is_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    logger.info(
        f"[UNIVERSE] Loading universe profile: {universe_name}, "
        f"quote={quote}, max_symbols={max_symbols}, testnet={is_testnet}"
    )
    
    try:
        if universe_name in ("l1l2-top", "l1l2volume", "l1l2-vol"):
            symbols = get_l1l2_top_by_volume(quote=quote, max_symbols=max_symbols)
        
        elif universe_name == "megacap":
            symbols = get_megacap_universe(quote=quote, max_symbols=max_symbols)
        
        elif universe_name in ("all-usdt", "all", "full"):
            symbols = get_all_usdt_universe(quote=quote, max_symbols=max_symbols)
        
        else:
            logger.warning(
                f"[UNIVERSE] Unknown profile '{universe_name}', "
                f"falling back to 'megacap'"
            )
            symbols = get_megacap_universe(quote=quote, max_symbols=max_symbols)
        
        # Filter for testnet if needed
        if is_testnet:
            original_count = len(symbols)
            symbols = [s for s in symbols if s in TESTNET_WHITELIST]
            if len(symbols) < original_count:
                logger.warning(
                    f"[UNIVERSE] Testnet filter: {original_count} → {len(symbols)} symbols "
                    f"(removed {original_count - len(symbols)} invalid testnet symbols)"
                )
        
        logger.info(
            f"[UNIVERSE] Profile={universe_name}, requested={max_symbols}, "
            f"selected={len(symbols)}, testnet_filtered={is_testnet}"
        )
        
        return symbols
    
    except Exception as e:
        logger.error(
            f"[UNIVERSE] Failed to load universe '{universe_name}': {e}, "
            f"using minimal fallback"
        )
        # Minimal fallback to prevent crashes
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]


def save_universe_snapshot(
    symbols: List[str],
    mode: str,
    qt_universe: Optional[str] = None,
    qt_max_symbols: Optional[int] = None,
    snapshot_path: str = "/app/data/universe_snapshot.json"
) -> None:
    """Save universe snapshot for debugging and reproducibility.
    
    Args:
        symbols: List of symbols in the universe
        mode: "explicit" or "dynamic"
        qt_universe: Universe profile name (if dynamic mode)
        qt_max_symbols: Max symbols limit (if dynamic mode)
        snapshot_path: Path to save JSON snapshot
    """
    try:
        snapshot_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "qt_universe": qt_universe,
            "qt_max_symbols": qt_max_symbols,
            "symbol_count": len(symbols),
            "symbols": symbols
        }
        
        # Ensure directory exists
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(snapshot_path, "w") as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"[UNIVERSE] Snapshot written to {snapshot_path}")
    
    except Exception as e:
        logger.warning(f"[UNIVERSE] Failed to write snapshot: {e}")


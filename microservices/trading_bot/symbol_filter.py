"""
Symbol Filter - Velger top coins basert på volum og kategori
"""
import aiohttp
import logging
from typing import List, Dict, Set

logger = logging.getLogger(__name__)

# Mainnet, Layer 1 og Layer 2 blockchain prosjekter
# Ekskluderer meme coins, utility tokens, wrapped assets, og leveraged tokens
MAINNET_L1_L2_COINS = {
    # Major Layer 1
    "BTC", "ETH", "BNB", "SOL", "ADA", "AVAX", "DOT", "ATOM", "NEAR", "ALGO",
    "TON", "TRX", "XRP", "LTC", "ETC", "XLM", "FTM", "EGLD", "KAS", "HBAR",
    "VET", "ICP", "FIL", "XTZ", "EOS", "FLOW", "MINA", "APT", "SUI", "SEI",
    
    # Layer 2 & Scaling Solutions
    "MATIC", "ARB", "OP", "IMX", "METIS", "BOBA", "LRC", "STRK", "MANTA",
    
    # DeFi Infrastructure & Interoperability (considered mainnet)
    "LINK", "UNI", "AAVE", "MKR", "CRV", "SNX", "COMP", "SUSHI", "BAL",
    
    # Privacy & Special Purpose Chains
    "ZEC", "XMR", "DASH", "ZEN",
    
    # Polkadot/Cosmos Ecosystem (L1 parachains)
    "GLMR", "MOVR", "ASTR", "KAVA", "OSMO", "JUNO", "INJ",
    
    # Smart Contract Platforms
    "NEO", "QTUM", "WAVES", "ONT", "ZIL", "ICX", "IOST",
    
    # Newer L1s
    "APTOS", "SUI", "CFX", "CELO", "ROSE", "KAVA",
    
    # Gaming/Metaverse Infrastructure (L1/L2)
    "IMX", "SAND", "MANA", "AXS", "GALA", "ENJ",
    
    # DePIN / Infrastructure
    "FIL", "AR", "THETA", "GRT", "RENDER", "AKT"
}


async def fetch_top_symbols_by_volume(
    limit: int = 50,
    min_volume_usd: float = 10_000_000  # Min 10M USD 24h volum
) -> List[str]:
    """
    Hent top N symbols basert på 24h handelsvolum.
    Filtrer kun mainnet/L1/L2 coins.
    
    Args:
        limit: Antall symbols å returnere
        min_volume_usd: Minimum 24h volum i USD
        
    Returns:
        Liste med symbol names (f.eks. ["BTCUSDT", "ETHUSDT", ...])
    """
    try:
        logger.info(f"[SYMBOL-FILTER] Fetching top {limit} symbols by 24h volume...")
        
        # Hent 24h ticker data fra Binance Futures
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.error(f"[SYMBOL-FILTER] Failed to fetch tickers: HTTP {resp.status}")
                    return get_fallback_symbols(limit)
                
                tickers = await resp.json()
        
        # Filtrer og sorter
        filtered_tickers = []
        
        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            
            # Kun USDT perpetuals
            if not symbol.endswith("USDT"):
                continue
            
            # Ekstrakt base asset (f.eks. BTC fra BTCUSDT)
            base_asset = symbol.replace("USDT", "")
            
            # Skip leveraged tokens, inverse, og ikke-mainnet coins
            if any(skip in base_asset for skip in ["UP", "DOWN", "BEAR", "BULL", "1000"]):
                continue
            
            # Kun mainnet/L1/L2 coins
            if base_asset not in MAINNET_L1_L2_COINS:
                logger.debug(f"[SYMBOL-FILTER] Skipping {symbol} - not mainnet/L1/L2")
                continue
            
            # Parse volum
            try:
                volume_usd = float(ticker.get("quoteVolume", 0))
                price_change = float(ticker.get("priceChangePercent", 0))
                
                # Minimum volum requirement
                if volume_usd < min_volume_usd:
                    continue
                
                filtered_tickers.append({
                    "symbol": symbol,
                    "volume_usd": volume_usd,
                    "price_change_pct": price_change,
                    "base_asset": base_asset
                })
                
            except (ValueError, TypeError) as e:
                logger.debug(f"[SYMBOL-FILTER] Failed to parse {symbol}: {e}")
                continue
        
        # Sorter etter volum (høyest først)
        filtered_tickers.sort(key=lambda x: x["volume_usd"], reverse=True)
        
        # Ta top N
        top_symbols = [t["symbol"] for t in filtered_tickers[:limit]]
        
        logger.info(
            f"[SYMBOL-FILTER] ✅ Selected {len(top_symbols)} symbols from {len(filtered_tickers)} candidates"
        )
        
        # Log top 10 for debugging
        for i, ticker in enumerate(filtered_tickers[:10], 1):
            logger.info(
                f"[SYMBOL-FILTER]   #{i} {ticker['symbol']}: "
                f"${ticker['volume_usd']/1e6:.1f}M volume, "
                f"{ticker['price_change_pct']:+.2f}% change"
            )
        
        return top_symbols if top_symbols else get_fallback_symbols(limit)
        
    except Exception as e:
        logger.error(f"[SYMBOL-FILTER] Error fetching symbols: {e}", exc_info=True)
        return get_fallback_symbols(limit)


def get_fallback_symbols(limit: int = 50) -> List[str]:
    """
    Fallback liste hvis API fetch feiler.
    Statisk liste med høyt volum mainnet/L1/L2 coins.
    """
    fallback = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "ETCUSDT",
        "NEARUSDT", "ALGOUSDT", "FILUSDT", "ICPUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "INJUSDT", "LDOUSDT", "STXUSDT",
        "TIAUSDT", "SUIUSDT", "RENDERUSDT", "TRXUSDT", "TONUSDT",
        "FTMUSDT", "XLMUSDT", "HBARUSDT", "XTZUSDT", "VETUSDT",
        "GRTUSDT", "AAVEUSDT", "MKRUSDT", "SUSHIUSDT", "COMPUSDT",
        "SNXUSDT", "CRVUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
        "ENJUSDT", "CHZUSDT", "THETAUSDT", "KSMUSDT", "GALAUSDT"
    ]
    
    logger.warning(f"[SYMBOL-FILTER] Using fallback symbol list ({len(fallback[:limit])} symbols)")
    return fallback[:limit]


async def refresh_symbols_periodically(
    bot,
    refresh_interval_hours: int = 6
):
    """
    Oppdater symbol liste periodisk - PRIORITET: AI policy, deretter volum.
    
    Args:
        bot: SimpleTradingBot instance
        refresh_interval_hours: Hvor ofte listen skal oppdateres
    """
    import asyncio
    import json
    
    while bot.running:
        try:
            # Vent refresh interval
            await asyncio.sleep(refresh_interval_hours * 3600)
            
            logger.info("[SYMBOL-FILTER] 🔄 Refreshing symbol list...")
            
            # PRIORITY 1: Read from AI policy
            new_symbols = []
            if bot.event_bus and "redis" in bot.event_bus:
                try:
                    policy_data = await bot.event_bus["redis"].hget("quantum:policy:current", "universe_symbols")
                    if policy_data:
                        # Decode bytes to string if needed
                        if isinstance(policy_data, bytes):
                            policy_data = policy_data.decode('utf-8')
                        policy_symbols = json.loads(policy_data)
                        if isinstance(policy_symbols, list) and len(policy_symbols) > 0:
                            new_symbols = policy_symbols
                            logger.info(f"[SYMBOL-FILTER] ✅ Using {len(new_symbols)} symbols from AI policy")
                except Exception as e:
                    logger.warning(f"[SYMBOL-FILTER] ⚠️  Could not read AI policy: {e}")
            
            # FALLBACK: Hent basert på volum
            if not new_symbols:
                logger.info("[SYMBOL-FILTER] Falling back to volume-based selection...")
                new_symbols = await fetch_top_symbols_by_volume(limit=50)
            
            if new_symbols and new_symbols != bot.symbols:
                old_count = len(bot.symbols)
                bot.symbols = new_symbols
                logger.info(
                    f"[SYMBOL-FILTER] ✅ Updated symbol list: "
                    f"{old_count} → {len(new_symbols)} symbols"
                )
            else:
                logger.info("[SYMBOL-FILTER] Symbol list unchanged")
                
        except Exception as e:
            logger.error(f"[SYMBOL-FILTER] Error refreshing symbols: {e}", exc_info=True)
            await asyncio.sleep(3600)  # Retry etter 1 time ved feil


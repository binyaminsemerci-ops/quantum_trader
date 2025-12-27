"""
Trading Universe - Comprehensive list of tradeable assets
Organized by category for easier management.

This module defaults to USDT tickers but can re-quote all symbols
to a different quote currency (e.g., USDC) based on environment
variables from `config.config`.
"""

from __future__ import annotations

import os
from typing import Iterable, List
try:
    from config.config import DEFAULT_QUOTE as _DEFAULT_QUOTE
except Exception:
    _DEFAULT_QUOTE = os.environ.get("DEFAULT_QUOTE", "USDT")

# Layer 1 Blockchains (Main networks)
LAYER1_COINS = [
    "BTCUSDT",    # Bitcoin
    "ETHUSDT",    # Ethereum
    "BNBUSDT",    # BNB Chain
    "SOLUSDT",    # Solana
    "ADAUSDT",    # Cardano
    "AVAXUSDT",   # Avalanche
    "DOTUSDT",    # Polkadot
    "MATICUSDT",  # Polygon (now POL but MATIC ticker)
    "ATOMUSDT",   # Cosmos
    "NEARUSDT",   # NEAR Protocol
    "ALGOUSDT",   # Algorand
    "ICPUSDT",    # Internet Computer
    "APTUSDT",    # Aptos
    "SUIUSDT",    # Sui
    "INJUSDT",    # Injective
    "SEIUSDT",    # Sei
    "FTMUSDT",    # Fantom
    "HBARUSDT",   # Hedera
    "VETUSDT",    # VeChain
    "QNTUSDT",    # Quant
    "TONUSDT",    # TON
    "TRXUSDT",    # Tron
]

# Layer 2 Solutions
LAYER2_COINS = [
    "ARBUSDT",    # Arbitrum
    "OPUSDT",     # Optimism
    "METISUSDT",  # Metis
    "MANTAUSDT",  # Manta
    "STRKUSDT",   # Starknet
    "IMXUSDT",    # Immutable X
]

# DeFi Protocols
DEFI_COINS = [
    "UNIUSDT",    # Uniswap
    "LINKUSDT",   # Chainlink
    "AAVEUSDT",   # Aave
    "MKRUSDT",    # Maker
    "COMPUSDT",   # Compound
    "CRVUSDT",    # Curve
    "LDOUSDT",    # Lido DAO
    "SNXUSDT",    # Synthetix
    "PENDLEUSDT", # Pendle
    "RUNEUSDT",   # THORChain
    "1INCHUSDT",  # 1inch
]

# Smart Contract Platforms & Infrastructure
INFRASTRUCTURE_COINS = [
    "RENDERUSDT", # Render Network
    "FILUSDT",    # Filecoin
    "ARUSDT",     # Arweave
    "GRTUSDT",    # The Graph
    "OCEANUSDT",  # Ocean Protocol
]

# Meme & Community Coins (High volatility - good for trading)
MEME_COINS = [
    "DOGEUSDT",   # Dogecoin
    "SHIBUSDT",   # Shiba Inu
    "PEPEUSDT",   # Pepe
    "WIFUSDT",    # Dogwifhat
    "BONKUSDT",   # Bonk
    "FLOKIUSDT",  # Floki
]

# Gaming & Metaverse
GAMING_COINS = [
    "AXSUSDT",    # Axie Infinity
    "SANDUSDT",   # The Sandbox
    "MANAUSDT",   # Decentraland
    "APEUSDT",    # ApeCoin
    "GALAUSDT",   # Gala
    "IMXUSDT",    # Immutable X
]

# AI & Data
AI_COINS = [
    "FETUSDT",    # Fetch.ai
    "AGIXUSDT",   # SingularityNET
    "RNDRENDERUSDT", # Render (duplicate, use RENDERUSDT)
    "OCEANGPTUSDT",  # Ocean (duplicate)
]

# Stablecoins & RWA
STABLECOIN_COINS = [
    # Note: Most stablecoins don't have good trading signals
    # But some algorithmic ones can be traded
    "DAIUSDT",    # DAI (if available)
]

# High Volume Altcoins (Good liquidity)
HIGH_VOLUME_ALTS = [
    "XRPUSDT",    # Ripple
    "LTCUSDT",    # Litecoin
    "BCHUSDT",    # Bitcoin Cash
    "ETCUSDT",    # Ethereum Classic
    "XLMUSDT",    # Stellar
    "XMRUSDT",    # Monero (if available)
]

# New & Trending (Update periodically)
TRENDING_COINS = [
    "WLDUSDT",    # Worldcoin
    "TIAUSDT",    # Celestia
    "PYRUSDT",    # Pyth Network
    "DYMUSDT",    # Dymension
    "JUPUSDT",    # Jupiter
    "WUSDT",      # Wormhole
]

# Build complete trading universe
ALL_COINS = list(set(
    LAYER1_COINS +
    LAYER2_COINS +
    DEFI_COINS +
    INFRASTRUCTURE_COINS +
    MEME_COINS +
    GAMING_COINS +
    HIGH_VOLUME_ALTS +
    TRENDING_COINS
))

# Default set (most liquid, best for starting)
DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
    "UNIUSDT", "ATOMUSDT", "ARBUSDT", "OPUSDT", "DOGEUSDT",
    "SHIBUSDT", "PEPEUSDT", "APTUSDT", "SUIUSDT", "INJUSDT",
]

# Aggressive universe (including high-volatility meme coins)
AGGRESSIVE_UNIVERSE = DEFAULT_UNIVERSE + MEME_COINS + TRENDING_COINS

# Conservative universe (only major L1s)
CONSERVATIVE_UNIVERSE = LAYER1_COINS[:10]

def apply_quote(symbols: Iterable[str], quote: str | None = None) -> List[str]:
    """Convert symbols to the desired quote currency by suffix replacement.

    - Replaces trailing USDT/USDC with the target `quote`.
    - Leaves symbols unchanged if they already match the target quote
      or do not end with a known quote suffix.
    """
    q = (quote or _DEFAULT_QUOTE).upper()
    out: List[str] = []
    for s in symbols:
        if s.endswith(q):
            out.append(s)
            continue
        if s.endswith("USDT") or s.endswith("USDC"):
            out.append(s[:-4] + q)
        else:
            # Unknown suffix: leave as-is
            out.append(s)
    return out


# Get universe by risk profile
def get_trading_universe(profile: str = "default", *, quote: str | None = None) -> list:
    """
    Get trading universe based on risk profile
    
    Args:
        profile: 'default', 'aggressive', 'conservative', 'all'
        quote: Optional quote override (e.g., 'USDC'). If not provided,
               uses DEFAULT_QUOTE from config.
    """
    if profile == "aggressive":
        base = AGGRESSIVE_UNIVERSE
    elif profile == "conservative":
        base = CONSERVATIVE_UNIVERSE
    elif profile == "all":
        base = ALL_COINS
    else:
        base = DEFAULT_UNIVERSE

    return apply_quote(base, quote=quote or _DEFAULT_QUOTE)


if __name__ == "__main__":
    print(f"Total coins available: {len(ALL_COINS)}")
    print(f"Default universe: {len(DEFAULT_UNIVERSE)} coins")
    print(f"Aggressive universe: {len(AGGRESSIVE_UNIVERSE)} coins")
    print(f"Conservative universe: {len(CONSERVATIVE_UNIVERSE)} coins")
    print("\nAll coins:")
    for coin in sorted(ALL_COINS):
        print(f"  {coin}")

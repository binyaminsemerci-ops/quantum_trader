"""
Market configuration for multi-market trading on Binance.
Supports SPOT (USDC), FUTURES (USDC), MARGIN (USDC/USDT), CROSS-MARGIN (USDC/USDT).
"""

from typing import Dict, List
import logging

# Layer 1 & Layer 2 tokens with high volume (top 50)
LAYER1_LAYER2_TOKENS = [
    # Layer 1 Blockchains
    "BTC",
    "ETH",
    "BNB",
    "ADA",
    "SOL",
    "AVAX",
    "DOT",
    "ATOM",
    "NEAR",
    "ALGO",
    "XTZ",
    "FTM",
    "LUNA",
    "EGLD",
    "FLOW",
    "ICP",
    "KSM",
    "ONE",
    "ROSE",
    "KAVA",
    # Layer 2 & Scaling Solutions
    "MATIC",
    "LRC",
    "OP",
    "ARB",
    "IMX",
    "METIS",
    "BOBA",
    "CELR",
    # High Volume DeFi & Infrastructure
    "UNI",
    "AAVE",
    "SUSHI",
    "CRV",
    "COMP",
    "MKR",
    "YFI",
    "1INCH",
    "BAL",
    "SNX",
    "LINK",
    "GRT",
    "BAND",
    "API3",
    "ALPHA",
    "RUNE",
    "CAKE",
    "XVS",
    # High Volume Meme & Community Tokens
    "DOGE",
    "SHIB",
    "PEPE",
    "FLOKI",
    "BONK",
    "WIF",
    "MEME",
    # Stablecoins & Wrapped Assets
    "WBTC",
    "STETH",
    "RETH",
    "FRXETH",
]

# Market types and their base currencies
MARKET_CONFIGS = {
    "SPOT": {
        "base_currencies": ["USDC"],
        "leverage": 1,
        "margin_enabled": False,
        "description": "Spot trading with USDC",
    },
    "FUTURES": {
        "base_currencies": ["USDC"],
        "leverage": 30,  # Aggressive leverage for all trades
        "margin_enabled": True,
        "description": "USDC-M Futures trading",
    },
    "MARGIN": {
        "base_currencies": ["USDC", "USDT"],
        "leverage": 3,  # Typical margin leverage
        "margin_enabled": True,
        "description": "Isolated margin trading",
    },
    "CROSS_MARGIN": {
        "base_currencies": ["USDC", "USDT"],
        "leverage": 5,  # Cross margin leverage
        "margin_enabled": True,
        "description": "Cross margin trading",
    },
}


def get_trading_pairs(market_type: str = "SPOT") -> List[str]:
    """Get all valid trading pairs for the specified market type."""
    pairs = []

    if market_type not in MARKET_CONFIGS:
        raise ValueError(f"Unsupported market type: {market_type}")

    base_currencies = MARKET_CONFIGS[market_type]["base_currencies"]
    if not isinstance(base_currencies, list):
        raise TypeError(
            f"Expected list for base_currencies, got {type(base_currencies)}"
        )

    for token in LAYER1_LAYER2_TOKENS:
        for base in base_currencies:
            if token != base:  # Don't create USDC/USDC pairs
                if market_type == "FUTURES":
                    pairs.append(f"{token}USDC")  # Futures use USDC-M contracts
                else:
                    pairs.append(f"{token}{base}")

    return pairs


def get_market_config(market_type: str) -> Dict:
    """Get configuration for specific market type."""
    return MARKET_CONFIGS.get(market_type, MARKET_CONFIGS["SPOT"])


def is_layer1_layer2_token(symbol: str) -> bool:
    """Check if symbol is a Layer 1 or Layer 2 token."""
    # Extract base token from trading pair
    base_token = symbol.replace("USDC", "").replace("USDT", "").replace("BUSD", "")
    return base_token in LAYER1_LAYER2_TOKENS


def get_volume_weighted_pairs(market_type: str = "SPOT", top_n: int = 50) -> List[str]:
    """Get top N volume-weighted trading pairs for the market type."""
    all_pairs = get_trading_pairs(market_type)

    # For now, return the first top_n pairs
    # In production, this would query Binance 24hr ticker API to get real volume data
    return all_pairs[:top_n]


def get_optimal_market_for_token(token: str, strategy: str = "volume") -> str:
    """Determine optimal market type for a given token based on strategy."""

    # High volume tokens typically better on futures for leverage
    high_volume_tokens = ["BTC", "ETH", "BNB", "SOL", "ADA", "MATIC", "DOGE", "SHIB"]

    if strategy == "volume" and token in high_volume_tokens:
        return "FUTURES"
    elif strategy == "conservative":
        return "SPOT"
    elif strategy == "aggressive":
        return "CROSS_MARGIN"
    else:
        return "SPOT"


# Risk parameters per market type
RISK_CONFIGS = {
    "SPOT": {
        "max_position_size": 0.10,  # 10% of portfolio per position
        "stop_loss": 0.05,  # 5% stop loss
        "take_profit": 0.15,  # 15% take profit
        "max_daily_trades": 20,
    },
    "FUTURES": {
        "max_position_size": 0.25,  # 25% of portfolio MARGIN per position (4 max positions)
        "stop_loss": 0.03,  # 3% stop loss (margin-based with 30x leverage)
        "take_profit": 0.10,  # 10% take profit (margin-based)
        "max_daily_trades": 15,
        "max_leverage": 30,
    },
    "MARGIN": {
        "max_position_size": 0.08,  # 8% of portfolio per position
        "stop_loss": 0.04,  # 4% stop loss
        "take_profit": 0.12,  # 12% take profit
        "max_daily_trades": 18,
    },
    "CROSS_MARGIN": {
        "max_position_size": 0.06,  # 6% of portfolio per position
        "stop_loss": 0.035,  # 3.5% stop loss
        "take_profit": 0.12,  # 12% take profit
        "max_daily_trades": 12,
    },
}


def get_risk_config(market_type: str) -> Dict:
    """Get risk configuration for specific market type."""
    return RISK_CONFIGS.get(market_type, RISK_CONFIGS["SPOT"])


# Logging configuration
logging.getLogger(__name__).info(
    f"Loaded {len(LAYER1_LAYER2_TOKENS)} Layer 1/2 tokens for multi-market trading"
)

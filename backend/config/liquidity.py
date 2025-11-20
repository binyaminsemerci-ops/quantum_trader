"""Liquidity selection configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_tuple(value: str | None, *, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if not value:
        return default
    entries = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    return entries or default


@dataclass(frozen=True)
class LiquidityConfig:
    """Runtime configuration for liquidity universe selection."""

    universe_max: int = 200  # Increased from 100 to ensure we get 100+ after filtering
    selection_min: int = 10
    selection_max: int = 50  # Increased from 20 to allow more positions
    min_quote_volume: float = 500_000.0  # Lowered to include more pairs
    score_momentum_weight: float = 0.15
    binance_weight: float = 0.7
    coingecko_weight: float = 0.3
    stable_quote_assets: Tuple[str, ...] = ("USDT", "USDC", "BUSD")  # Added BUSD
    blacklist_suffixes: Tuple[str, ...] = ("UP", "DOWN", "BULL", "BEAR")  # Removed PERP to allow perpetuals
    liquidity_weight: float = 0.55
    model_weight: float = 0.45
    model_sell_threshold: float = 0.55
    max_per_base: int = 2  # Allow both USDT and USDC for same coin
    # Futures/Margin specific
    market_type: str = "spot"  # spot | usdm_perp | coinm_perp
    margin_mode: str = "cross"  # cross | isolated
    default_leverage: int = 5    # applied when placing futures orders (target)


def load_liquidity_config() -> LiquidityConfig:
    """Load liquidity configuration from environment variables."""

    universe_max = _parse_int(os.getenv("QT_LIQUIDITY_UNIVERSE_MAX"), default=200)
    selection_min = _parse_int(os.getenv("QT_LIQUIDITY_SELECTION_MIN"), default=10)
    selection_max = _parse_int(os.getenv("QT_LIQUIDITY_SELECTION_MAX"), default=50)
    min_quote_volume = _parse_float(
        os.getenv("QT_LIQUIDITY_MIN_QUOTE_VOLUME"),
        default=500_000.0,
    )
    score_momentum_weight = _parse_float(
        os.getenv("QT_LIQUIDITY_MOMENTUM_WEIGHT"),
        default=0.15,
    )
    binance_weight = _parse_float(
        os.getenv("QT_LIQUIDITY_BINANCE_WEIGHT"),
        default=0.7,
    )
    coingecko_weight = _parse_float(
        os.getenv("QT_LIQUIDITY_COINGECKO_WEIGHT"),
        default=0.3,
    )
    # Allow quick disable of Coingecko provider via env toggle
    if os.getenv("QUANTUM_TRADER_DISABLE_COINGECKO") == "1":
        coingecko_weight = 0.0
    liquidity_weight = _parse_float(
        os.getenv("QT_LIQUIDITY_LIQ_WEIGHT"),
        default=0.55,
    )
    model_weight = _parse_float(
        os.getenv("QT_LIQUIDITY_MODEL_WEIGHT"),
        default=0.45,
    )
    model_sell_threshold = _parse_float(
        os.getenv("QT_LIQUIDITY_MODEL_SELL_THRESHOLD"),
        default=0.55,
    )
    stable_quote_assets = _parse_tuple(
        os.getenv("QT_LIQUIDITY_STABLE_QUOTES"),
        default=("USDT", "USDC", "BUSD"),
    )
    blacklist_suffixes = _parse_tuple(
        os.getenv("QT_LIQUIDITY_BLACKLIST_SUFFIXES"),
        default=("UP", "DOWN", "BULL", "BEAR"),
    )
    max_per_base = _parse_int(
        os.getenv("QT_LIQUIDITY_MAX_PER_BASE"),
        default=2,
    )
    if max_per_base < 0:
        max_per_base = 0
    market_type = os.getenv("QT_MARKET_TYPE", "spot").strip().lower()
    if market_type not in {"spot", "usdm_perp", "coinm_perp"}:
        market_type = "spot"
    margin_mode = os.getenv("QT_MARGIN_MODE", "cross").strip().lower()
    if margin_mode not in {"cross", "isolated"}:
        margin_mode = "cross"
    try:
        default_leverage = int(os.getenv("QT_DEFAULT_LEVERAGE", "30"))
    except ValueError:
        default_leverage = 30
    default_leverage = max(1, min(default_leverage, 125))  # Binance limits
    weight_sum = liquidity_weight + model_weight
    if weight_sum <= 0:
        liquidity_weight, model_weight = 0.55, 0.45
    else:
        liquidity_weight = liquidity_weight / weight_sum
        model_weight = model_weight / weight_sum

    selection_max = max(selection_min, selection_max)

    return LiquidityConfig(
        universe_max=universe_max,
        selection_min=selection_min,
        selection_max=selection_max,
        min_quote_volume=min_quote_volume,
        score_momentum_weight=score_momentum_weight,
        binance_weight=binance_weight,
        coingecko_weight=coingecko_weight,
        stable_quote_assets=stable_quote_assets,
        blacklist_suffixes=blacklist_suffixes,
        liquidity_weight=liquidity_weight,
        model_weight=model_weight,
        model_sell_threshold=model_sell_threshold,
        max_per_base=max_per_base,
        market_type=market_type,
        margin_mode=margin_mode,
        default_leverage=default_leverage,
    )


__all__ = ["LiquidityConfig", "load_liquidity_config"]

"""Config package shim.

This file makes ``config`` an explicit package and re-exports the
convenience symbols from ``config.config`` so existing imports continue to
work and mypy has a consistent module mapping.
"""

from __future__ import annotations

from .config import (
    DEFAULT_EXCHANGE,
    DEFAULT_QUOTE,
    FUTURES_QUOTE,
    load_config,
    make_pair,
    masked_config_summary,
    settings,
    get_qt_symbols,
    get_qt_universe,
    get_qt_max_symbols,
)

# Re-export liquidity config for backward compatibility
try:
    from backend.config.liquidity import LiquidityConfig, load_liquidity_config
except ImportError:
    # Fallback if backend not importable
    LiquidityConfig = None  # type: ignore
    load_liquidity_config = None  # type: ignore

__all__ = [
    "DEFAULT_EXCHANGE",
    "DEFAULT_QUOTE",
    "FUTURES_QUOTE",
    "load_config",
    "make_pair",
    "masked_config_summary",
    "settings",
    "get_qt_symbols",
    "get_qt_universe",
    "get_qt_max_symbols",
    "LiquidityConfig",
    "load_liquidity_config",
]

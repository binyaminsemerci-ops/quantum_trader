"""Minimal runtime configuration for tests and CI.

This file provides a tiny, safe implementation that other modules import
from. It intentionally keeps behavior simple: values are read from
environment variables with sensible defaults so tests and CI don't need a
complex external config system.
"""

from __future__ import annotations

import os

from typing import Dict, Any
from types import SimpleNamespace


# A simple default exchange name used by the exchange factory when no
# explicit name is provided. Keep this stable across tests.
DEFAULT_EXCHANGE: str = os.environ.get("DEFAULT_EXCHANGE", "binance")


# Common quote symbols used across the codebase. Keep these simple so tests
# and CI which import them from `config.config` don't fail.
DEFAULT_QUOTE: str = os.environ.get("DEFAULT_QUOTE", "USDT")
FUTURES_QUOTE: str = os.environ.get("FUTURES_QUOTE", DEFAULT_QUOTE)


def make_pair(base: str, quote: str | None = None) -> str:
    """Build a simple market pair string used by helpers and tests.

    The implementation is intentionally minimal: it concatenates base+quote
    so callers can import and use it without requiring exchange SDKs.
    """
    q = quote or FUTURES_QUOTE
    return f"{base}{q}"


def load_config() -> Any:
    """Return a minimal settings object (SimpleNamespace) for tests and CI.

    Many modules in the repository access configuration attributes (e.g.
    `cfg.binance_api_key`) so returning a SimpleNamespace preserves
    attribute-style access while keeping the implementation tiny.
    """
    ns = SimpleNamespace(
        DEFAULT_EXCHANGE=os.environ.get("DEFAULT_EXCHANGE", DEFAULT_EXCHANGE),
        QUANTUM_TRADER_DATABASE_URL=os.environ.get(
            "QUANTUM_TRADER_DATABASE_URL", "sqlite:///./db.sqlite3"
        ),
        DEFAULT_QUOTE=os.environ.get("DEFAULT_QUOTE", DEFAULT_QUOTE),
        FUTURES_QUOTE=os.environ.get("FUTURES_QUOTE", FUTURES_QUOTE),
        binance_api_key=os.environ.get("BINANCE_API_KEY"),
        binance_api_secret=os.environ.get("BINANCE_API_SECRET"),
        coinbase_api_key=os.environ.get("COINBASE_API_KEY"),
        coinbase_api_secret=os.environ.get("COINBASE_API_SECRET"),
    )

    # Attach convenience helpers to the namespace so tests that import
    # these via `config.config` still find them.
    setattr(ns, "make_pair", make_pair)

    return ns


def masked_config_summary(cfg: Any) -> Dict[str, Any]:
    """Return a tiny, masked summary of secrets suitable for health endpoints.

    The real application might omit or redact secrets; for CI/tests we
    provide a deterministic (non-secret) structure.
    """
    return {
        "has_binance_keys": bool(
            getattr(cfg, "binance_api_key", None)
            and getattr(cfg, "binance_api_secret", None)
        ),
        "has_coinbase_keys": bool(
            getattr(cfg, "coinbase_api_key", None)
            and getattr(cfg, "coinbase_api_secret", None)
        ),
    }


# Convenience variable expected by some modules (and by the shim).
settings: Any = load_config()


__all__ = ["DEFAULT_EXCHANGE", "load_config", "settings"]

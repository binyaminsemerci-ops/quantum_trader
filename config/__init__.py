"""Config package initializer.

Ensures the local `config` directory is treated as a standard package rather than
being shadowed by any similarly named third-party distribution.  Intentionally
kept minimal.
"""

# Optionally, explicit re-exports could be added here, e.g.:
# from .config import *  # noqa: F401,F403
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
)

__all__ = [
    "DEFAULT_EXCHANGE",
    "DEFAULT_QUOTE",
    "FUTURES_QUOTE",
    "load_config",
    "make_pair",
    "masked_config_summary",
    "settings",
]

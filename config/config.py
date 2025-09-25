"""Minimal runtime configuration for tests and CI.

This file provides a tiny, safe implementation that other modules import
from. It intentionally keeps behavior simple: values are read from
environment variables with sensible defaults so tests and CI don't need a
complex external config system.
"""
from __future__ import annotations

import os
from typing import Dict, Any


# A simple default exchange name used by the exchange factory when no
# explicit name is provided. Keep this stable across tests.
DEFAULT_EXCHANGE: str = os.environ.get("DEFAULT_EXCHANGE", "binance")


def load_config() -> Dict[str, Any]:
	"""Return a minimal settings dict for use in tests and runtime.

	We keep the shape generic (dict) so callers can access keys directly.
	"""
	return {
		"DEFAULT_EXCHANGE": os.environ.get("DEFAULT_EXCHANGE", DEFAULT_EXCHANGE),
		"QUANTUM_TRADER_DATABASE_URL": os.environ.get("QUANTUM_TRADER_DATABASE_URL", "sqlite:///./db.sqlite3"),
	}


# Convenience variable expected by some modules (and by the shim).
settings: Dict[str, Any] = load_config()


__all__ = ["DEFAULT_EXCHANGE", "load_config", "settings"]

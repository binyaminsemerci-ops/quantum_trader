"""Lightweight Binance price stream manager (mini real-time ticker).

Provides a shared in-memory price cache updated from Binance WebSocket
streams. Falls back gracefully if API keys are missing or connection fails.

Usage:
  from backend.services.price_stream import ensure_price_stream, get_price_snapshot
  ensure_price_stream(symbols)
  prices = get_price_snapshot()

Thread-safety: python-binance callbacks run in thread(s). We protect the cache
with a simple threading.Lock for consistency (reads are cheap).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - runtime env dependent
    from binance import ThreadedWebsocketManager  # type: ignore
except Exception:  # pragma: no cover
    ThreadedWebsocketManager = None  # type: ignore

from config.config import load_config

logger = logging.getLogger(__name__)

_price_cache: Dict[str, Dict[str, float]] = {}
_orderbook_cache: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
_cache_lock = threading.Lock()
_started = False
_twm = None  # type: ignore[var-annotated]
_subscribed: set[str] = set()
_last_error: str | None = None
_mock_mode: bool = False


def _on_symbol_ticker(msg):  # type: ignore
    # Example msg keys: {'e': '24hrTicker', 'E': 123456789, 's': 'BNBBTC', 'c': '0.0025', ...}
    try:
        symbol = msg.get("s")
        price = msg.get("c")  # last price
        if symbol and price:
            with _cache_lock:
                _price_cache[symbol] = {
                    "price": float(price),
                    "ts": time.time(),
                }
    except Exception as e:  # pragma: no cover
        logger.debug("Ticker callback parse error: %s", e)


def _mock_prices(symbols: Iterable[str]):
    now = time.time()
    import math
    import random

    for sym in symbols:
        h = hash(sym) % 10_000
        base = 10 + (h % 500)  # base value
        wiggle = math.sin(now / 5 + h) * (base * 0.01)
        price = base + wiggle + random.random()
        _price_cache[sym] = {"price": round(price, 4), "ts": now}


def ensure_price_stream(symbols: Iterable[str], limit: int = 20):
    """Ensure websocket manager is running and subscribed to given symbols.

    - Starts TWM once if API keys available.
    - Subscribes (up to limit) to symbol ticker sockets.
    - Ignores silently if python-binance not installed or keys missing.
    """
    global _started, _twm, _last_error
    cfg = load_config()
    if not getattr(cfg, "binance_api_key", None) or not getattr(
        cfg, "binance_api_secret", None
    ):
        # Activate mock mode for demo environments
        _last_error = "missing_api_keys"
        global _mock_mode
        _mock_mode = True
        _mock_prices(symbols)
        return
    if ThreadedWebsocketManager is None:
        _last_error = "twm_unavailable"
        return
    if not _started:
        try:
            _twm = ThreadedWebsocketManager(
                api_key=cfg.binance_api_key,
                api_secret=cfg.binance_api_secret,
                testnet=bool(cfg.binance_use_testnet),
            )
            _twm.start()
            _started = True
            logger.info(
                "Price stream manager started (testnet=%s)", cfg.binance_use_testnet
            )
        except Exception as e:  # pragma: no cover
            _last_error = f"start_failed:{e}"
            logger.error("Failed to start ThreadedWebsocketManager: %s", e)
            return

    if not _twm:  # safety
        return

    count = 0
    for sym in symbols:
        if count >= limit:
            break
        if sym in _subscribed:
            continue
        try:
            # Use symbol ticker socket (24hrTicker) for simplicity
            _twm.start_symbol_ticker_socket(callback=_on_symbol_ticker, symbol=sym)
            _subscribed.add(sym)
            count += 1
        except Exception as e:  # pragma: no cover
            logger.debug("Subscription failed for %s: %s", sym, e)
    # If mock mode previously enabled we keep updating those too
    if _mock_mode:
        _mock_prices(symbols)


def get_price_snapshot() -> Dict[str, Dict[str, float]]:
    with _cache_lock:
        return {k: v.copy() for k, v in _price_cache.items()}


def get_orderbook_snapshot(symbol: str) -> Dict[str, List[Tuple[float, float]]]:
    # Placeholder: future real depth integration; for now derive pseudo depth from price
    with _cache_lock:
        price = _price_cache.get(symbol, {}).get("price")
    if price is None:
        return {"bids": [], "asks": []}
    # create synthetic 5-level book
    levels = 5
    spread = price * 0.001
    bids = [(round(price - i * spread, 4), 1 + i * 0.5) for i in range(1, levels + 1)]
    asks = [(round(price + i * spread, 4), 1 + i * 0.45) for i in range(1, levels + 1)]
    return {"bids": bids, "asks": asks}


def get_last_error() -> str | None:
    return _last_error


__all__ = [
    "ensure_price_stream",
    "get_price_snapshot",
    "get_last_error",
    "get_orderbook_snapshot",
]

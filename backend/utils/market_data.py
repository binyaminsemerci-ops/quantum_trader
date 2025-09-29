
"""Market data adapters with optional ccxt integration.

If ccxt is available and ``ENABLE_LIVE_MARKET_DATA=1`` (or true) the helpers
fetch real OHLC candles via the configured exchange. Otherwise they fall back
to deterministic demo data so the application keeps functioning offline and in
CI.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
import logging

from config.config import load_config, DEFAULT_QUOTE
from backend.routes.settings import SETTINGS

try:  # pragma: no cover - ccxt is optional
    import ccxt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ccxt not installed
    ccxt = None

logger = logging.getLogger(__name__)


def _normalize_symbol(symbol: str, quote: str) -> str:
    if '/' in symbol:
        return symbol
    if symbol.upper().endswith(quote.upper()):
        base = symbol[: -len(quote)]
        return f"{base}/{quote}"
    return f"{symbol}/{quote}"


def _demo_candles(symbol: str, limit: int) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    base = 100.0 + (hash(symbol) % 50)
    candles: List[Dict[str, Any]] = []
    for i in range(limit):
        ts = now - timedelta(minutes=(limit - i))
        open_p = base + (i * 0.1) + (0.5 * (i % 3))
        close_p = open_p + ((-1) ** i) * (0.5 * ((i % 5) / 5.0))
        high_p = max(open_p, close_p) + 0.4
        low_p = min(open_p, close_p) - 0.4
        volume = 10 + (i % 7)
        candles.append(
            {
                "time": ts.isoformat(),
                "open": round(open_p, 3),
                "high": round(high_p, 3),
                "low": round(low_p, 3),
                "close": round(close_p, 3),
                "volume": volume,
            }
        )
    return candles


def fetch_recent_candles(symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
    cfg = load_config()
    enable_live = bool(SETTINGS.get('ENABLE_LIVE_MARKET_DATA', getattr(cfg, 'enable_live_market_data', False)))
    if enable_live and ccxt is not None:
        try:
            exchange_id = getattr(cfg, 'default_exchange', 'binance')
            exchange_class = getattr(ccxt, exchange_id)
            params = {
                "enableRateLimit": True,
                "timeout": getattr(cfg, 'ccxt_timeout', 10000),
            }
            api_key = getattr(cfg, 'binance_api_key', None)
            api_secret = getattr(cfg, 'binance_api_secret', None)
            if api_key and api_secret:
                params["apiKey"] = api_key
                params["secret"] = api_secret
            exchange = exchange_class(params)
            market = _normalize_symbol(symbol, getattr(cfg, 'default_quote', DEFAULT_QUOTE))
            timeframe = getattr(cfg, 'ccxt_timeframe', '1m')
            ohlcv = exchange.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
            candles = []
            for ts, open_p, high_p, low_p, close_p, volume in ohlcv:
                candles.append(
                    {
                        "time": datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat(),
                        "open": float(open_p),
                        "high": float(high_p),
                        "low": float(low_p),
                        "close": float(close_p),
                        "volume": float(volume),
                    }
                )
            if candles:
                return candles
        except Exception as exc:  # pragma: no cover - network/exchange specific
            logger.warning("Falling back to demo candles: %s", exc)
    return _demo_candles(symbol, limit)


def _demo_signals(symbol: str, limit: int, profile: str = "mixed") -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    signals: List[Dict[str, Any]] = []
    for i in range(limit):
        ts = now - timedelta(minutes=5 * i)
        if profile == "left":
            side = "sell" if i % 3 else "buy"
        elif profile == "right":
            side = "buy" if i % 3 else "sell"
        else:
            side = "buy" if i % 2 == 0 else "sell"
        signals.append(
            {
                "id": f"demo-{symbol}-{i}",
                "timestamp": ts,
                "symbol": symbol,
                "side": side,
                "score": round(0.5 + ((-1) ** i) * 0.05, 3),
                "confidence": 0.5 + ((i % 5) / 10),
                "details": {"source": "demo", "note": f"mock signal #{i}"},
            }
        )
    return signals


def fetch_recent_signals(symbol: str, limit: int = 20, profile: str = "mixed") -> List[Dict[str, Any]]:
    cfg = load_config()
    enable_live = bool(SETTINGS.get('ENABLE_LIVE_MARKET_DATA', getattr(cfg, 'enable_live_market_data', False)))
    if enable_live:
        candles = fetch_recent_candles(symbol, limit + 1)
        if len(candles) > 1:
            signals: List[Dict[str, Any]] = []
            prev_close = candles[0]["close"]
            for candle in candles[1:]:
                close = candle["close"]
                side = "buy" if close >= prev_close else "sell"
                magnitude = abs(close - prev_close)
                score = max(0.0, min(1.0, magnitude / max(prev_close, 1.0)))
                signals.append(
                    {
                        "id": f"live-{symbol}-{candle['time']}",
                        "timestamp": datetime.fromisoformat(candle["time"]),
                        "symbol": symbol,
                        "side": side,
                        "score": round(score, 3),
                        "confidence": min(1.0, 0.4 + score),
                        "details": {"source": "momentum", "note": "derived from OHLC momentum"},
                    }
                )
                prev_close = close
            if signals:
                return signals[-limit:]
    return _demo_signals(symbol, limit, profile)

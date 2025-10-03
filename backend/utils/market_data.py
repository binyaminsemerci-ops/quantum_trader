
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

from config.config import load_config, settings
from backend.routes.settings import SETTINGS
from backend.utils.exchanges import resolve_exchange_name, resolve_credentials
from ai_engine.agents.xgb_agent import make_default_agent

# Cache for ML agent to avoid repeated disk loads every poll
_CACHED_AGENT = None  # type: ignore
_AGENT_FAILED = False

# Cache for signals to avoid slow ML operations on every dashboard poll
_SIGNALS_CACHE = {}  # symbol -> (timestamp, signals_list)
_SIGNALS_CACHE_TTL = 30.0  # Cache signals for 30 seconds

def _get_agent_once():
    global _CACHED_AGENT, _AGENT_FAILED
    if _CACHED_AGENT is not None:
        return _CACHED_AGENT
    if _AGENT_FAILED:
        return None
    try:
        _CACHED_AGENT = make_default_agent()
        return _CACHED_AGENT
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load ML agent (cached): %s", exc)
        _AGENT_FAILED = True
        return None

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
    enable_live = bool(
        SETTINGS.get('ENABLE_LIVE_MARKET_DATA', getattr(cfg, 'enable_live_market_data', False))
    )
    if enable_live and ccxt is not None:
        exchange_name = resolve_exchange_name(getattr(cfg, 'default_exchange', None))
        try:
            exchange_class = getattr(ccxt, exchange_name)
        except AttributeError:
            logger.warning("Unknown ccxt exchange '%s'; falling back to demo data", exchange_name)
        else:
            params: Dict[str, Any] = {
                "enableRateLimit": True,
                "timeout": getattr(cfg, 'ccxt_timeout', 10000),
                "sandbox": False,  # Use live public API
            }
            api_key, api_secret = resolve_credentials(exchange_name, None, None)
            # Only add credentials if we have BOTH key and secret (for trading, not market data)
            if api_key and api_secret:
                params["apiKey"] = api_key
                params["secret"] = api_secret
            exchange = None
            try:
                exchange = exchange_class(params)
                market = _normalize_symbol(symbol, getattr(cfg, 'default_quote', settings.default_quote))
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
            finally:
                if exchange is not None:
                    try:
                        exchange.close()
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass
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
    # Check cache first - return cached signals if fresh (within TTL)
    import time
    now = time.time()
    cache_key = f"{symbol}:{limit}:{profile}"

    if cache_key in _SIGNALS_CACHE:
        cached_time, cached_signals = _SIGNALS_CACHE[cache_key]
        if (now - cached_time) < _SIGNALS_CACHE_TTL:
            return cached_signals

    cfg = load_config()
    enable_live = bool(SETTINGS.get("ENABLE_LIVE_MARKET_DATA", getattr(cfg, "enable_live_market_data", False)))

    if enable_live:
        agent = _get_agent_once()
        if agent is not None:
            candles = fetch_recent_candles(symbol, limit + 200)
            if len(candles) >= 20:
                history = min(240, max(60, len(candles)))
                start_idx = max(0, len(candles) - limit)
                signals: List[Dict[str, Any]] = []
                for idx in range(start_idx, len(candles)):
                    window_start = max(0, idx - history + 1)
                    window = candles[window_start : idx + 1]
                    result = agent.predict_for_symbol(window)
                    action = str(result.get("action", "HOLD")).upper()
                    raw_score = float(result.get("score", 0.0) or 0.0)
                    score = abs(raw_score)
                    if action not in {"BUY", "SELL"}:
                        continue
                    ts_str = candles[idx].get("time")
                    try:
                        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
                    except Exception:
                        ts = datetime.now(timezone.utc)
                    side = "buy" if action == "BUY" else "sell"
                    confidence = min(1.0, 0.4 + score)
                    signals.append(
                        {
                            "id": f"model-{symbol}-{idx}",
                            "timestamp": ts,
                            "symbol": symbol,
                            "side": side,
                            "score": round(score, 4),
                            "confidence": round(confidence, 4),
                            "details": {
                                "source": "model",
                                "note": "xgb_agent",
                                "raw_action": action,
                                "raw_score": raw_score,
                            },
                        }
                    )
                if signals:
                    # Cache the ML signals for future requests
                    _SIGNALS_CACHE[cache_key] = (now, signals[-limit:])
                    return signals[-limit:]

    # Generate demo signals and cache them too
    demo_signals = _demo_signals(symbol, limit, profile)
    _SIGNALS_CACHE[cache_key] = (now, demo_signals)
    return demo_signals

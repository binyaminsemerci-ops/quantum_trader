"""Exchange-facing routes using the pluggable exchange clients.

All endpoints return deterministic demo data when live market access is
disabled. When `ENABLE_LIVE_MARKET_DATA` is true (and credentials are
configured) they call through to the configured exchange adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.routes.settings import SETTINGS
from backend.utils.exchanges import (
    ExchangeClient,
    get_exchange_client,
    resolve_exchange_name,
)
from config.config import load_config

router = APIRouter()


def _live_enabled() -> bool:
    cfg = load_config()
    override = SETTINGS.get("ENABLE_LIVE_MARKET_DATA")
    if override is not None:
        return str(override).lower() in {"1", "true", "yes", "on"}
    return bool(getattr(cfg, "enable_live_market_data", False))


def _determine_exchange(name: Optional[str]) -> str:
    cfg = load_config()
    preferred = name or getattr(cfg, "default_exchange", None)
    return resolve_exchange_name(preferred)


def _demo_spot() -> Dict[str, Any]:
    return {"asset": "USDC", "free": 1000.0, "source": "demo"}


def _demo_futures() -> Dict[str, Any]:
    return {"asset": "USDT", "balance": 0.0, "source": "demo"}


def _demo_trades(symbol: str, limit: int) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "symbol": symbol,
            "qty": "0.01",
            "price": "100.0",
            "side": "buy" if i % 2 == 0 else "sell",
            "time": now,
        }
        for i in range(limit)
    ]


def _client_or_raise(name: str) -> ExchangeClient:
    try:
        return get_exchange_client(name)
    except Exception as exc:  # pragma: no cover - adapter construction errors
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/spot-balance")
def spot_balance(exchange: Optional[str] = Query(None)) -> Dict[str, Any]:
    name = _determine_exchange(exchange)
    if not _live_enabled():
        return {"exchange": name, "balance": _demo_spot()}
    client = _client_or_raise(name)
    try:
        balance = client.spot_balance()
    except Exception as exc:  # pragma: no cover - adapter specific
        raise HTTPException(
            status_code=503, detail=f"Unable to fetch spot balance: {exc}"
        ) from exc
    return {"exchange": name, "balance": balance}


@router.get("/futures-balance")
def futures_balance(exchange: Optional[str] = Query(None)) -> Dict[str, Any]:
    name = _determine_exchange(exchange)
    if not _live_enabled():
        return {"exchange": name, "balance": _demo_futures()}
    client = _client_or_raise(name)
    try:
        balance = client.futures_balance()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail=f"Unable to fetch futures balance: {exc}"
        ) from exc
    return {"exchange": name, "balance": balance}


@router.get("/recent-trades")
def recent_trades(
    symbol: str = Query("BTCUSDT", min_length=4),
    limit: int = Query(5, ge=1, le=100),
    exchange: Optional[str] = Query(None),
) -> Dict[str, Any]:
    name = _determine_exchange(exchange)
    if not _live_enabled():
        return {"exchange": name, "trades": _demo_trades(symbol, limit)}
    client = _client_or_raise(name)
    try:
        trades = client.fetch_recent_trades(symbol, limit=limit)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail=f"Unable to fetch trades: {exc}"
        ) from exc
    return {"exchange": name, "symbol": symbol, "trades": trades[:limit]}


@router.get("/server-time")
def server_time(exchange: Optional[str] = Query(None)) -> Dict[str, Any]:
    name = _determine_exchange(exchange)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return {"exchange": name, "serverTime": now_ms}

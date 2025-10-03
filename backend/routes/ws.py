from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
import time
from sqlalchemy import select, func
from typing import Any, cast

from backend.database import session_scope, Trade, EquityPoint
from backend.utils.pnl import calculate_pnl, calculate_pnl_per_symbol
from backend.utils.risk import calculate_risk
from backend.utils.analytics import calculate_analytics
from backend.routes.portfolio import get_portfolio, get_market_overview
from config.config import load_config
from backend.routes.signals import recent_signals  # reuse logic

try:
    from backend.services.binance_trading import get_trading_engine  # type: ignore
except Exception:  # pragma: no cover

    def get_trading_engine():  # type: ignore
        raise RuntimeError("Trading engine unavailable")
from backend.utils.market_data import fetch_recent_signals
from backend.services.price_stream import (
    ensure_price_stream,
    get_price_snapshot,
    get_last_error,
    get_orderbook_snapshot,
)

_WS_START_TIME = time.time()

router = APIRouter()

# Simple in-memory caches
_price_cache: dict[str, dict[str, Any]] = {}
_last_price_fetch: float = 0.0


async def _fetch_prices(symbols: list[str]):
    """Fetch latest prices from Binance client if available, else skip.
    Updates _price_cache with {'price': float, 'ts': epoch_seconds} per symbol.
    Rate-limited to at most once per 2s.
    """
    global _last_price_fetch
    now = time.time()
    if now - _last_price_fetch < 2:  # rate limit
        return
    try:
        engine = get_trading_engine()
        client = getattr(engine, "client", None)
        if client is None:
            return
        # Use get_symbol_ticker for each symbol (python-binance call)
        for sym in symbols:
            try:
                ticker = client.get_symbol_ticker(symbol=sym)
                price = (
                    float(ticker.get("price"))
                    if ticker and ticker.get("price")
                    else None
                )
                if price is not None:
                    _price_cache[sym] = {"price": price, "ts": now}
            except Exception:
                # ignore symbol-level failures
                continue
        _last_price_fetch = now
    except Exception:
        pass


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    """Unified real-time dashboard stream.

    Sends a payload with keys expected by the simplified frontend dashboard:
      - systemStatus
      - marketData
      - portfolio
      - signals
      - trades
      - chartData
      - stats (detailed metrics)
    """
    await websocket.accept()
    try:
        while True:
            with session_scope() as session:
                # Core aggregate stats
                total_trades = (
                    session.scalar(select(func.count(cast(Any, Trade.id)))) or 0
                )
                avg_price = (
                    session.scalar(select(func.avg(cast(Any, Trade.price)))) or 0.0
                )
                active_symbols = (
                    session.scalar(
                        select(func.count(func.distinct(cast(Any, Trade.symbol))))
                    )
                    or 0
                )

                # Recent trades
                trades = [
                    {
                        "id": trade.id,
                        "timestamp": (
                            trade.timestamp.isoformat() if trade.timestamp else None
                        ),
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "qty": float(trade.qty) if trade.qty is not None else None,
                        "price": (
                            float(trade.price) if trade.price is not None else None
                        ),
                    }
                    for trade in session.execute(
                        select(Trade)
                        .order_by(cast(Any, Trade.timestamp).desc())
                        .limit(30)
                    ).scalars()
                ]

                # Equity curve -> normalized to chartData (timestamp, price)
                eq_rows = list(
                    session.execute(
                        select(EquityPoint).order_by(cast(Any, EquityPoint.date).asc())
                    ).scalars()
                )
                chart_points = [
                    {
                        "timestamp": point.date.isoformat() if point.date else None,
                        "price": (
                            float(point.equity) if point.equity is not None else None
                        ),
                    }
                    for point in eq_rows
                ]

                # Portfolio + Market overview (reuse existing async route logic via direct call)
                # Call async functions properly
                portfolio_data = await get_portfolio(db=session)
                market_overview = await get_market_overview(db=session)

                # Signals (fast path direct util for performance)
                try:
                    signal_records = fetch_recent_signals(
                        symbol="BTCUSDT", limit=15, profile="mixed"
                    )
                    signals_payload = []
                    for item in signal_records:
                        ts = item.get("timestamp")
                        if hasattr(ts, "isoformat"):
                            ts_val = ts.isoformat()
                        else:
                            ts_val = ts
                        signals_payload.append(
                            {
                                "symbol": item.get("symbol", "BTCUSDT"),
                                "side": item.get("side", "BUY"),
                                "confidence": round(
                                    float(
                                        item.get("score", item.get("confidence", 0.5))
                                    )
                                    * 100,
                                    2,
                                ),
                                "ts": ts_val,
                            }
                        )
                except Exception:
                    signals_payload = []

            # Build system status lightweight (avoid importing full route logic again)
            uptime_seconds = round(time.time() - _WS_START_TIME, 2)
            cfg = None
            try:
                cfg = load_config()
            except Exception:
                pass
            system_status = {
                "service": "quantum_trader_core",
                "uptime_seconds": uptime_seconds,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "binance_testnet": (
                    bool(getattr(cfg, "binance_use_testnet", False)) if cfg else None
                ),
                "has_binance_keys": (
                    bool(
                        getattr(cfg, "binance_api_key", None)
                        and getattr(cfg, "binance_api_secret", None)
                    )
                    if cfg
                    else None
                ),
            }

            # Trading status + market ticks
            try:
                engine = get_trading_engine()
                trading_status = engine.get_trading_status()
                symbols = engine.get_trading_symbols()[
                    :16
                ]  # extend list (limit for payload)
            except Exception:
                trading_status = {
                    "is_running": False,
                    "error": "trading engine not active",
                }
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

            # Start/ensure streaming subscriptions (non-blocking)
            try:
                ensure_price_stream(symbols)
            except Exception:
                pass

            # Attempt REST fetch only if streaming cache is sparse
            price_snapshot = get_price_snapshot()
            if (
                len([s for s in symbols if s in price_snapshot]) < 3
            ):  # fallback to REST warming
                await _fetch_prices(symbols)

            # Build market ticks, using streaming/REST cache; fallback to last trade price if cache empty
            market_ticks = []
            try:
                with session_scope() as s2:
                    for sym in symbols:
                        cache_entry = price_snapshot.get(sym) or _price_cache.get(sym)
                        price = cache_entry.get("price") if cache_entry else None
                        if price is None:  # fallback query
                            last_trade = s2.execute(
                                select(Trade)
                                .where(Trade.symbol == sym)
                                .order_by(cast(Any, Trade.timestamp).desc())
                                .limit(1)
                            ).scalar_one_or_none()
                            if last_trade and last_trade.price is not None:
                                price = float(last_trade.price)
                        market_ticks.append(
                            {
                                "symbol": sym,
                                "price": price,
                                "age_ms": (
                                    (time.time() - cache_entry["ts"]) * 1000
                                    if cache_entry and cache_entry.get("ts")
                                    else None
                                ),
                                "src": (
                                    "stream"
                                    if sym in price_snapshot
                                    else ("rest" if price is not None else "fallback")
                                ),
                            }
                        )
            except Exception:
                pass
            except Exception:
                trading_status = {
                    "is_running": False,
                    "error": "trading engine not active",
                }

            # Synthetic equity fallback if no chart points: build from cumulative PnL timeline (simplistic)
            persist_equity = False
            if not chart_points:
                # Use trades ordered by time to accumulate pnl progressively
                with session_scope() as s3:
                    trade_rows = list(
                        s3.execute(
                            select(Trade).order_by(cast(Any, Trade.timestamp).asc())
                        ).scalars()
                    )
                cumulative = 0.0
                synthetic = []
                for tr in trade_rows:
                    if tr.side.upper() == "SELL":
                        cumulative += (tr.qty or 0) * (tr.price or 0)
                    else:
                        cumulative -= (tr.qty or 0) * (tr.price or 0)
                    synthetic.append(
                        {
                            "timestamp": (
                                tr.timestamp.isoformat() if tr.timestamp else None
                            ),
                            "price": round(cumulative, 2),
                        }
                    )
                chart_points = synthetic[-200:]  # limit
                # Persist last synthetic point as EquityPoint every loop if any trades exist
                if synthetic:
                    persist_equity = True

            # Daily PnL change (24h): compare cumulative pnl now vs pnl from trades older than 24h
            pnl_now = calculate_pnl()
            pnl_24h = 0.0
            try:
                cutoff = __import__("datetime").datetime.utcnow() - __import__(
                    "datetime"
                ).timedelta(hours=24)
                with session_scope() as s4:
                    old_rows = s4.execute(
                        select(
                            Trade.side, Trade.qty, Trade.price, Trade.timestamp
                        ).where(cast(Any, Trade.timestamp) <= cutoff)
                    ).all()
                pnl_24h = 0.0
                for side, qty, price, _ts in old_rows:
                    if side.upper() == "SELL":
                        pnl_24h += qty * price
                    else:
                        pnl_24h -= qty * price
            except Exception:
                pass
            daily_change = round(pnl_now - pnl_24h, 2)
            starting_equity = (
                getattr(cfg, "starting_equity", 10000.0) if cfg else 10000.0
            )
            pnl_percent = (
                round((pnl_now / starting_equity) * 100, 2) if starting_equity else 0.0
            )
            daily_change_percent = (
                round(((daily_change) / starting_equity) * 100, 2)
                if starting_equity
                else 0.0
            )

            # Persist synthetic equity (best-effort; ignore failures)
            if persist_equity:
                try:
                    last = chart_points[-1]
                    with session_scope() as ps:
                        import datetime

                        ep = EquityPoint(
                            date=datetime.datetime.utcnow(),
                            equity=float(last["price"] or 0),
                        )
                        ps.add(ep)
                except Exception:
                    pass

            latency_flag = any(
                t.get("age_ms", 0) and t.get("age_ms", 0) > 10_000 for t in market_ticks
            )
            primary_symbol = symbols[0] if symbols else "BTCUSDT"
            orderbook = get_orderbook_snapshot(primary_symbol)

            payload = {
                "systemStatus": system_status,
                "marketData": market_overview,
                "portfolio": portfolio_data,
                "signals": signals_payload,
                "trades": trades,
                "chartData": chart_points,
                "trading": trading_status,
                "marketTicks": market_ticks,
                "orderbook": {"symbol": primary_symbol, **orderbook},
                "stats": {
                    "total_trades": total_trades,
                    "avg_price": round(avg_price, 2),
                    "active_symbols": active_symbols,
                    "pnl": calculate_pnl(),
                    "pnl_per_symbol": calculate_pnl_per_symbol(),
                    "daily_pnl_change": daily_change,
                    "pnl_percent": pnl_percent,
                    "daily_pnl_change_percent": daily_change_percent,
                    "latency_high": latency_flag,
                    "risk": calculate_risk(),
                    "analytics": calculate_analytics(),
                },
                "stream_meta": {
                    "stream_error": get_last_error(),
                    "price_symbols": len(market_ticks),
                },
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        # Graceful disconnect logging
        print("[ws] Client disconnected /ws/dashboard")


_chat_clients: set[WebSocket] = set()


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    _chat_clients.add(websocket)
    try:
        await websocket.send_text(
            json.dumps({"system": "welcome", "message": "Chat connected"})
        )
        while True:
            msg = await websocket.receive_text()
            payload = {
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "message": msg[:500],
            }
            # Broadcast
            dead = []
            for ws in _chat_clients:
                try:
                    await ws.send_text(json.dumps(payload))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _chat_clients.discard(ws)
    except WebSocketDisconnect:
        _chat_clients.discard(websocket)
        print("[ws] Client disconnected /ws/chat")


@router.websocket("/ws/enhanced-data")
async def enhanced_data_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for enhanced multi-source data."""
    from backend.routes.enhanced_api import enhanced_manager

    await enhanced_manager.connect(websocket)

    try:
        # Send initial data
        from backend.routes.external_data import enhanced_market_data

        initial_data = await enhanced_market_data(["BTC", "ETH", "ADA", "SOL"])
        await websocket.send_text(
            json.dumps(
                {
                    "type": "enhanced_data_update",
                    "data": initial_data,
                    "timestamp": time.time(),
                }
            )
        )

        # Keep connection alive and listen for client messages
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        enhanced_manager.disconnect(websocket)

import asyncio
import hmac
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError

try:
    from database import Base, TradeLog, get_db  # type: ignore
except ImportError:  # pragma: no cover - fallback for package layout
    from backend.database import Base, TradeLog, get_db  # type: ignore

try:
    from backend.utils.admin_auth import get_configured_admin_token
    from backend.utils.admin_events import AdminEvent, record_admin_event
except ImportError:  # pragma: no cover - fallback for package layout
    from utils.admin_auth import get_configured_admin_token  # type: ignore
    from utils.admin_events import AdminEvent, record_admin_event  # type: ignore

try:
    from backend.utils.pnl import calculate_pnl, calculate_pnl_per_symbol
    from backend.utils.risk import calculate_risk
    from backend.utils.analytics import calculate_analytics
except ImportError:  # pragma: no cover - fallback for package layout
    from utils.pnl import calculate_pnl, calculate_pnl_per_symbol  # type: ignore
    from utils.risk import calculate_risk  # type: ignore
    from utils.analytics import calculate_analytics  # type: ignore


logger = logging.getLogger(__name__)

router = APIRouter()


def _build_dashboard_payload(session) -> Dict[str, Any]:
    try:
        Base.metadata.create_all(bind=session.get_bind())
    except Exception as exc:  # pragma: no cover - defensive guard for schema setup
        logger.warning("Failed to ensure dashboard tables exist: %s", exc)

    total_trades = session.query(func.count(TradeLog.id)).scalar() or 0
    avg_price = session.query(func.avg(TradeLog.price)).scalar() or 0.0
    active_symbols = (
        session.query(func.count(func.distinct(TradeLog.symbol))).scalar() or 0
    )

    recent_trades = (
        session.query(TradeLog)
        .order_by(TradeLog.timestamp.desc())
        .limit(20)
        .all()
    )
    trade_payload = [
        {
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "qty": trade.qty,
            "price": trade.price,
            "status": trade.status,
            "reason": trade.reason,
            "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
        }
        for trade in recent_trades
    ]

    recent_logs = (
        session.query(TradeLog)
        .order_by(TradeLog.timestamp.desc())
        .limit(50)
        .all()
    )
    log_payload = [
        {
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "symbol": log.symbol,
            "side": log.side,
            "qty": log.qty,
            "price": log.price,
            "status": log.status,
            "reason": log.reason,
        }
        for log in recent_logs
    ]

    try:
        chart_rows = session.execute(
            text("SELECT date, equity FROM equity_curve ORDER BY date ASC")
        ).mappings()
        chart_payload = [dict(row) for row in chart_rows]
    except SQLAlchemyError:
        chart_payload = []

    return {
        "stats": {
            "total_trades": total_trades,
            "avg_price": round(avg_price or 0, 2),
            "active_symbols": active_symbols,
            "pnl": calculate_pnl(),
            "pnl_per_symbol": calculate_pnl_per_symbol(),
            "risk": calculate_risk(),
            "analytics": calculate_analytics(),
        },
        "trades": trade_payload,
        "logs": log_payload,
        "chart": chart_payload,
    }


async def _ensure_admin_access(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    expected_token = get_configured_admin_token()
    if not expected_token:
        return {
            "path": websocket.url.path,
            "client": _format_client(websocket),
        }

    # Check token in header (preferred) or query string (browser fallback)
    provided_token = websocket.headers.get("x-admin-token")
    if not provided_token:
        # Fallback: Check query string for browser WebSocket clients
        query_params = dict(websocket.query_params)
        provided_token = query_params.get("token")
    
    details = {
        "path": websocket.url.path,
        "client": _format_client(websocket),
    }

    if provided_token is None:
        record_admin_event(
            AdminEvent.ADMIN_AUTH_MISSING,
            success=False,
            details={**details, "surface": "dashboard_stream"},
        )
        await websocket.accept()
        await websocket.close(code=4401, reason="Missing admin token")
        raise WebSocketDisconnect(code=4401, reason="Missing admin token")

    if not hmac.compare_digest(provided_token, expected_token):
        record_admin_event(
            AdminEvent.ADMIN_AUTH_INVALID,
            success=False,
            details={**details, "surface": "dashboard_stream"},
        )
        await websocket.accept()
        await websocket.close(code=4403, reason="Invalid admin token")
        raise WebSocketDisconnect(code=4403, reason="Invalid admin token")

    return details


def _format_client(websocket: WebSocket) -> Optional[str]:
    client = websocket.client
    if client is None:
        return None
    host = getattr(client, "host", None)
    port = getattr(client, "port", None)
    if host and port:
        return f"{host}:{port}"
    if host:
        return host
    return None


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    auth_details = await _ensure_admin_access(websocket)
    if auth_details is None:
        return

    await websocket.accept()
    record_admin_event(
        AdminEvent.DASHBOARD_STREAM,
        success=True,
        details={**auth_details, "status": "connected"},
    )

    db_iter = get_db()
    session = next(db_iter)
    try:
        while True:
            try:
                payload = _build_dashboard_payload(session)
            except SQLAlchemyError as exc:
                logger.error("Failed to build dashboard snapshot: %s", exc)
                record_admin_event(
                    AdminEvent.DASHBOARD_STREAM,
                    success=False,
                    details={
                        **auth_details,
                        "error": "database_error",
                        "message": str(exc),
                    },
                )
                await websocket.close(code=1011, reason="Database error")
                break
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Unexpected dashboard streaming error")
                record_admin_event(
                    AdminEvent.DASHBOARD_STREAM,
                    success=False,
                    details={
                        **auth_details,
                        "error": "internal_error",
                        "message": str(exc),
                    },
                )
                await websocket.close(code=1011, reason="Internal server error")
                break

            # Wrap payload in Dashboard V3 event format
            event = {
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "payload": payload
            }
            await websocket.send_text(json.dumps(event))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected")
    finally:
        db_iter.close()

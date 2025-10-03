from fastapi import APIRouter, Query, Depends, WebSocket, WebSocketDisconnect
from typing import Annotated, Any, cast
from sqlalchemy.orm import Session
import asyncio

from backend.database import get_session, TradeLog, session_scope

router = APIRouter()


@router.get("/trade_logs")
async def get_trade_logs(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    db: Session = Depends(get_session),
):
    """Return the most recent trade logs."""
    logs = db.query(TradeLog).order_by(cast(Any, TradeLog.id).desc()).limit(limit).all()
    return {
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "symbol": log.symbol,
                "side": log.side,
                "qty": log.qty,
                "price": log.price,
                "status": log.status,
                "reason": log.reason,
            }
            for log in logs
        ]
    }


@router.websocket("/ws/trade_logs")
async def trade_logs_ws(websocket: WebSocket, since_id: int = 0):
    """Stream new trade logs to connected clients.

    The endpoint polls the database for TradeLog rows with id > since_id and
    sends any new logs as a JSON array. Clients should reconnect on close.
    """
    await websocket.accept()
    last_id = int(since_id or 0)
    try:
        while True:
            out = []
            with session_scope() as session:
                rows = (
                    session.query(TradeLog)
                    .filter(TradeLog.id > last_id)
                    .order_by(TradeLog.id.asc())
                    .all()
                )
                for row in rows:
                    out.append(
                        {
                            "id": row.id,
                            "timestamp": (
                                row.timestamp.isoformat() if row.timestamp else None
                            ),
                            "symbol": row.symbol,
                            "side": row.side,
                            "qty": row.qty,
                            "price": row.price,
                            "status": row.status,
                            "reason": row.reason,
                        }
                    )
                    last_id = max(last_id, row.id)
            if out:
                await websocket.send_json(out)
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return

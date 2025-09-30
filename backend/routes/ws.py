from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
from sqlalchemy import select, func
from typing import Any, cast

from backend.database import session_scope, Trade, TradeLog, EquityPoint
from backend.utils.pnl import calculate_pnl, calculate_pnl_per_symbol
from backend.utils.risk import calculate_risk
from backend.utils.analytics import calculate_analytics

router = APIRouter()


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with session_scope() as session:
                total_trades = session.scalar(select(func.count(cast(Any, Trade.id)))) or 0
                avg_price = session.scalar(select(func.avg(cast(Any, Trade.price)))) or 0.0
                active_symbols = session.scalar(select(func.count(func.distinct(cast(Any, Trade.symbol))))) or 0

                trades = [
                    {
                        "id": trade.id,
                        "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "qty": trade.qty,
                        "price": trade.price,
                    }
                    for trade in session.execute(
                        select(Trade).order_by(cast(Any, Trade.timestamp).desc()).limit(20)
                    ).scalars()
                ]

                logs = [
                    {
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                        "symbol": log.symbol,
                        "side": log.side,
                        "qty": log.qty,
                        "price": log.price,
                        "status": log.status,
                        "reason": log.reason,
                    }
                    for log in session.execute(
                        select(TradeLog).order_by(cast(Any, TradeLog.timestamp).desc()).limit(50)
                    ).scalars()
                ]

                chart = [
                    {
                        "date": point.date.isoformat() if point.date else None,
                        "equity": point.equity,
                    }
                    for point in session.execute(
                        select(EquityPoint).order_by(cast(Any, EquityPoint.date).asc())
                    ).scalars()
                ]

            payload = {
                "stats": {
                    "total_trades": total_trades,
                    "avg_price": round(avg_price, 2),
                    "active_symbols": active_symbols,
                    "pnl": calculate_pnl(),
                    "pnl_per_symbol": calculate_pnl_per_symbol(),
                    "risk": calculate_risk(),
                    "analytics": calculate_analytics(),
                },
                "trades": trades,
                "logs": logs,
                "chart": chart,
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        print("?? Client disconnected")

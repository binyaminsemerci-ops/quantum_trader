from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.database import get_db
from backend.utils.pnl import calculate_pnl, calculate_pnl_per_symbol
from backend.utils.risk import calculate_risk
from backend.utils.analytics import calculate_analytics
import json
import asyncio

router = APIRouter()


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Use the SQLAlchemy session returned by get_db(); mypy can't know
            # the exact type so treat it as Any. For quick compatibility with
            # existing runtime code that expects a DB cursor, try to obtain a
            # raw connection or fallback to session.execute where possible.
            db = next(get_db())
            from typing import Any as _Any
            session: _Any = db

            try:
                # Prefer SQLAlchemy's execute() when available
                if hasattr(session, 'execute'):
                    total_trades = int(session.execute("SELECT COUNT(*) FROM trades").scalar() or 0)
                    avg_price = float(session.execute("SELECT AVG(price) FROM trades").scalar() or 0.0)
                    active_symbols = int(session.execute("SELECT COUNT(DISTINCT symbol) FROM trades").scalar() or 0)

                    trades_rows = list(session.execute("SELECT timestamp, symbol, side, qty, price FROM trades ORDER BY id DESC LIMIT 20"))
                    trades = [dict(r._mapping) for r in trades_rows]

                    logs_rows = list(session.execute("SELECT timestamp, symbol, side, qty, price, status, reason FROM trade_logs ORDER BY id DESC LIMIT 50"))
                    logs = [dict(r._mapping) for r in logs_rows]

                    chart_rows = list(session.execute("SELECT date, equity FROM equity_curve ORDER BY date ASC"))
                    chart = [dict(r._mapping) for r in chart_rows]
                else:
                    # Fallback for DB-API cursor-like sessions
                    cursor = session.cursor()
                    cursor.execute("SELECT COUNT(*) FROM trades")
                    total_trades = cursor.fetchone()[0]
                    cursor.execute("SELECT AVG(price) FROM trades")
                    avg_price = cursor.fetchone()[0] or 0
                    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM trades")
                    active_symbols = cursor.fetchone()[0]

                    cursor.execute(
                        "SELECT timestamp, symbol, side, qty, price FROM trades ORDER BY id DESC LIMIT 20"
                    )
                    trades = [
                        dict(zip([d[0] for d in cursor.description], row))
                        for row in cursor.fetchall()
                    ]

                    cursor.execute(
                        "SELECT timestamp, symbol, side, qty, price, status, reason FROM trade_logs ORDER BY id DESC LIMIT 50"
                    )
                    logs = [
                        dict(zip([d[0] for d in cursor.description], row))
                        for row in cursor.fetchall()
                    ]

                    cursor.execute("SELECT date, equity FROM equity_curve ORDER BY date ASC")
                    chart = [
                        dict(zip([d[0] for d in cursor.description], row))
                        for row in cursor.fetchall()
                    ]
            finally:
                try:
                    session.close()
                except Exception:
                    pass

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
        print("🔌 Client disconnected")

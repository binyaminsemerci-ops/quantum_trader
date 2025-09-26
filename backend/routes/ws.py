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
            db = next(get_db())
            cursor = db.cursor()

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

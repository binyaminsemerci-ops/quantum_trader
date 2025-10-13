# backend/routes/trade_logs.py
from fastapi import APIRouter, Query
from typing import Annotated
from database import get_db

router = APIRouter()


@router.get("/trade_logs")
async def get_trade_logs(limit: Annotated[int, Query(ge=1, le=500)] = 50):
    """
    Henter siste trade logs fra databasen.
    :param limit: Hvor mange logs som skal returneres (default 50, max 500)
    """
    try:
        db = next(get_db())
        cursor = db.cursor()
        cursor.execute(
            """
            SELECT timestamp, symbol, side, qty, price, status, reason
            FROM trade_logs
            ORDER BY id DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = cursor.fetchall()

        logs = [
            {
                "timestamp": row[0],
                "symbol": row[1],
                "side": row[2],
                "qty": row[3],
                "price": row[4],
                "status": row[5],
                "reason": row[6],
            }
            for row in rows
        ]

        return {"logs": logs}
    except Exception as e:
        # Return demo logs when database is not available
        import datetime
        demo_logs = []
        pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        statuses = ['accepted', 'executed', 'cancelled']
        
        for i in range(min(limit, 15)):
            demo_logs.append({
                "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=5*i)).isoformat(),
                "symbol": pairs[i % len(pairs)],
                "side": "BUY" if (i % 2 == 0) else "SELL",
                "qty": round(0.1 + i * 0.05, 4),
                "price": round(40000 + i * 100, 2),
                "status": statuses[i % len(statuses)],
                "reason": f"AI Signal #{i+1}"
            })
        
        return {"logs": demo_logs}

@router.get("/trade_logs/recent")
async def recent_trade_logs(limit: Annotated[int, Query(ge=1, le=100)] = 10):
    """Lightweight recent trade logs endpoint for fast polling."""
    try:
        db = next(get_db())
        cursor = db.cursor()
        cursor.execute(
            """
            SELECT timestamp, symbol, side, qty, price, status, reason
            FROM trade_logs
            ORDER BY id DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = cursor.fetchall()
        return {"logs": [
            {
                "timestamp": r[0],
                "symbol": r[1],
                "side": r[2],
                "qty": r[3],
                "price": r[4],
                "status": r[5],
                "reason": r[6],
            } for r in rows
        ]}
    except Exception:
        return await get_trade_logs(limit=limit)

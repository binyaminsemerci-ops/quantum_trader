# backend/routes/trade_logs.py
from fastapi import APIRouter, Query
from backend.database import get_db

router = APIRouter()


@router.get("/trade_logs")
async def get_trade_logs(limit: int = Query(50, ge=1, le=500)):
    """
    Henter siste trade logs fra databasen.
    :param limit: Hvor mange logs som skal returneres (default 50, max 500)
    """
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

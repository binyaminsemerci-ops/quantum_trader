from fastapi import APIRouter, Query
from backend.database import get_db

router = APIRouter()


@router.get("/candles")
def get_candles(
    symbol: str = Query(..., description="Trading symbol f.eks. BTCUSDT"),
    limit: int = Query(100, description="Antall candles som skal hentes"),
):
    """
    Returner OHLCV-data fra SQLite candles-tabellen.
    """

    db = next(get_db())
    cursor = db.cursor()

    cursor.execute(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (symbol, limit),
    )

    rows = cursor.fetchall()
    candles = [
        {
            "timestamp": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5],
        }
        for row in rows
    ]

    # Returner i kronologisk rekkefølge
    return {"symbol": symbol, "candles": list(reversed(candles))}

from typing import Dict, List, Annotated
from fastapi import APIRouter, Query
import datetime

router = APIRouter()


@router.get("/recent")
def recent_prices(symbol: str = "BTCUSDT", limit: Annotated[int, Query(50, ge=1, le=500)] = 50) -> List[Dict]:
    """Return a deterministic demo series of candles for the requested symbol.

    This endpoint is intentionally simple and deterministic so the frontend can
    display a demo price chart without external data.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    candles: List[Dict] = []
    base = 100.0 + (hash(symbol) % 50)
    for i in range(limit):
        t = (now - datetime.timedelta(minutes=(limit - i))).isoformat()
        # small deterministic walk using i
        open_p = base + (i * 0.1) + (0.5 * (i % 3))
        close_p = open_p + ((-1) ** i) * (0.5 * ((i % 5) / 5.0))
        high_p = max(open_p, close_p) + 0.4
        low_p = min(open_p, close_p) - 0.4
        volume = 10 + (i % 7)
        candles.append(
            {
                "time": t,
                "open": round(open_p, 3),
                "high": round(high_p, 3),
                "low": round(low_p, 3),
                "close": round(close_p, 3),
                "volume": volume,
            }
        )
    return candles

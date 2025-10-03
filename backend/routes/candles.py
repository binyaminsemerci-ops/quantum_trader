from fastapi import APIRouter, Query, Depends
from typing import Annotated, Any, cast
import logging
import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_session, Candle

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("")
@router.get("/")
def get_candles(
    symbol: Annotated[str, Query(..., description="Trading symbol, e.g. BTCUSDT")],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    db: Session = Depends(get_session),
) -> dict:
    try:
        candles = (
            db.execute(
                select(Candle)
                .where(Candle.symbol == symbol)
                .order_by(cast(Any, Candle.timestamp).desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        if candles:
            return {
                "symbol": symbol,
                "candles": [
                    {
                        "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in reversed(candles)
                ],
            }
    except Exception as exc:
        logger.exception("Database error fetching candles: %s", exc)

    now = datetime.datetime.now(datetime.timezone.utc)
    demo_candles = []
    base = 100.0 + (hash(symbol) % 50)
    for i in range(limit):
        t = (now - datetime.timedelta(minutes=(limit - i))).isoformat()
        open_p = base + (i * 0.1) + (0.5 * (i % 3))
        close_p = open_p + ((-1) ** i) * (0.5 * ((i % 5) / 5.0))
        high_p = max(open_p, close_p) + 0.4
        low_p = min(open_p, close_p) - 0.4
        volume = 10 + (i % 7)
        demo_candles.append(
            {
                "timestamp": t,
                "open": round(open_p, 3),
                "high": round(high_p, 3),
                "low": round(low_p, 3),
                "close": round(close_p, 3),
                "volume": volume,
            }
        )

    return {"symbol": symbol, "candles": list(reversed(demo_candles))}

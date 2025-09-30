from fastapi import APIRouter, Query, Depends
from typing import Annotated, Any, cast
from sqlalchemy.orm import Session

from backend.database import get_session, TradeLog

router = APIRouter()


@router.get("/trade_logs")
async def get_trade_logs(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    db: Session = Depends(get_session),
):
    """Return the most recent trade logs."""
    logs = (
        db.query(TradeLog)
        .order_by(cast(Any, TradeLog.id).desc())
        .limit(limit)
        .all()
    )
    return {
        "logs": [
            {
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

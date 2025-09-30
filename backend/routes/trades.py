from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Any, cast

from sqlalchemy.orm import Session

from backend.database import get_session, TradeLog, Trade

router = APIRouter()


class TradeCreate(BaseModel):
    symbol: str
    side: str
    qty: float
    price: float


@router.get("", response_model=List[dict])
async def get_trades(db: Session = Depends(get_session)):
    trades = db.query(TradeLog).order_by(cast(Any, TradeLog.id).desc()).all()
    return [
        {"id": t.id, "symbol": t.symbol, "side": t.side, "qty": t.qty, "price": t.price}
        for t in trades
    ]


@router.post("", status_code=200)
async def create_trade(payload: TradeCreate, db: Session = Depends(get_session)):
    entry = Trade(symbol=payload.symbol, side=payload.side, qty=payload.qty, price=payload.price)
    log = TradeLog(
        symbol=payload.symbol,
        side=payload.side,
        qty=payload.qty,
        price=payload.price,
        status="NEW",
    )
    db.add(entry)
    db.add(log)
    db.commit()
    db.refresh(entry)
    db.refresh(log)
    return {
        "id": entry.id,
        "symbol": entry.symbol,
        "side": entry.side,
        "qty": entry.qty,
        "price": entry.price,
    }


@router.get("/recent")
async def recent_trades(limit: int = 20):
    trades = []
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for i in range(limit):
        trades.append(
            {
                "id": f"t-{i}",
                "symbol": symbols[i % len(symbols)],
                "side": "BUY" if i % 2 == 0 else "SELL",
                "qty": round(0.01 * (i + 1), 4),
                "price": round(100 + i * 0.5, 2),
                "timestamp": i,
            }
        )
    return trades

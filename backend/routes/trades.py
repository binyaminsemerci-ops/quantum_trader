from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List

from backend.database import get_db, TradeLog

router = APIRouter()


class TradeCreate(BaseModel):
    symbol: str
    side: str
    qty: float
    price: float


@router.get("", response_model=List[dict])
async def get_trades(db=Depends(get_db)):
    # Return all trade logs as a list of dicts
    trades = list(db.query(TradeLog).all())
    return [
        {"id": t.id, "symbol": t.symbol, "side": t.side, "qty": t.qty, "price": t.price}
        for t in trades
    ]


@router.post("", status_code=200)
async def create_trade(payload: TradeCreate, db=Depends(get_db)):
    # Persist a new trade log and return its representation
    t = TradeLog(
        symbol=payload.symbol,
        side=payload.side,
        qty=payload.qty,
        price=payload.price,
        status="NEW",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return {
        "id": t.id,
        "symbol": t.symbol,
        "side": t.side,
        "qty": t.qty,
        "price": t.price,
    }



@router.get("/recent")
async def recent_trades(limit: int = 20):
    """Return a deterministic list of recent demo trades for frontend testing."""
    trades = []
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for i in range(limit):
        trades.append({
            "id": f"t-{i}",
            "symbol": symbols[i % len(symbols)],
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": round(0.01 * (i + 1), 4),
            "price": round(100 + i * 0.5, 2),
            "timestamp": i,
        })
    return trades

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy.exc import SQLAlchemyError
import logging

from backend.database import get_db, TradeLog

logger = logging.getLogger(__name__)

router = APIRouter()


class TradeCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol (e.g., BTCUSDT)")
    side: str = Field(..., pattern="^(BUY|SELL)$", description="Trade side: BUY or SELL")
    qty: float = Field(..., gt=0, description="Quantity must be positive")
    price: float = Field(..., gt=0, description="Price must be positive")


@router.get("", response_model=List[dict])
async def get_trades(db=Depends(get_db)):
    """Get all trade logs with proper error handling."""
    try:
        trades = list(db.query(TradeLog).all())
        logger.info(f"Retrieved {len(trades)} trades from database")
        
        return [
            {
                "id": t.id, 
                "symbol": t.symbol, 
                "side": t.side, 
                "qty": t.qty, 
                "price": t.price,
                "status": t.status,
                "timestamp": t.timestamp.isoformat() if t.timestamp else None
            }
            for t in trades
        ]
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving trades: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Unexpected error retrieving trades: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("", status_code=201)
async def create_trade(payload: TradeCreate, db=Depends(get_db)):
    """Create a new trade log with proper validation and error handling."""
    try:
        # Validate trade data
        if payload.symbol.upper() not in ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]:
            raise HTTPException(status_code=400, detail=f"Unsupported trading symbol: {payload.symbol}")
        
        if payload.side.upper() not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Side must be BUY or SELL")
            
        # Create trade log
        t = TradeLog(
            symbol=payload.symbol.upper(),
            side=payload.side.upper(), 
            qty=payload.qty,
            price=payload.price,
            status="NEW",
        )
        
        db.add(t)
        db.commit()
        db.refresh(t)
        
        logger.info(f"Created new trade: {t.id} - {t.side} {t.qty} {t.symbol} @ {t.price}")
        
        return {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": t.qty,
            "price": t.price,
            "status": t.status,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
            "message": "Trade created successfully"
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error creating trade: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error creating trade: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recent")
async def recent_trades(limit: int = 20):
    """Return a deterministic list of recent demo trades for frontend testing."""
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

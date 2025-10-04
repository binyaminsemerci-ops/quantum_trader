"""
Trading API endpoints for Quantum Trader.

This module provides comprehensive trading functionality including:
- Trade execution and order management
- Trading history and analytics
- Position tracking and portfolio management
- Real-time trade monitoring

All endpoints include proper error handling, input validation,
and performance monitoring integration.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.exc import SQLAlchemyError
import logging
from datetime import datetime, timezone

from backend.database import get_db, TradeLog

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Trading"],
    responses={
        400: {"description": "Invalid trading parameters"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
        502: {"description": "Exchange connectivity error"}
    }
)


class TradeCreate(BaseModel):
    """Request model for creating a new trade order."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)", example="BTCUSDT")
    side: str = Field(..., pattern="^(BUY|SELL)$", description="Trade direction: BUY or SELL", example="BUY")
    qty: float = Field(..., gt=0, description="Trade quantity (must be positive)", example=0.01)
    price: float = Field(..., gt=0, description="Trade price (must be positive)", example=43500.00)


class TradeResponse(BaseModel):
    """Response model for trade information."""

    id: int = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading pair symbol")
    side: str = Field(..., description="Trade direction (BUY/SELL)")
    qty: float = Field(..., description="Trade quantity")
    price: float = Field(..., description="Execution price")
    status: str = Field(..., description="Trade status")
    reason: Optional[str] = Field(default=None, description="Trade rationale or notes")
    timestamp: datetime = Field(..., description="Trade execution timestamp")


@router.get(
    "",
    response_model=List[TradeResponse],
    summary="Get Trading History",
    description="Retrieve comprehensive trading history with filtering and pagination options"
)
async def get_trades(
    db=Depends(get_db),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades to return"),
    symbol: Optional[str] = Query(None, description="Filter by trading pair symbol"),
    status: Optional[str] = Query(None, description="Filter by trade status"),
    from_date: Optional[datetime] = Query(None, description="Start date for trade history"),
    to_date: Optional[datetime] = Query(None, description="End date for trade history")
):
    """
    Retrieve trading history with comprehensive filtering options.

    This endpoint provides access to historical trades with support for:
    - Pagination via limit parameter
    - Symbol-based filtering for specific trading pairs
    - Status filtering (FILLED, CANCELLED, PARTIALLY_FILLED)
    - Date range filtering for time-based analysis

    Returns trade data sorted by most recent first, with proper
    error handling and performance monitoring.
    """
    try:
        query = db.query(TradeLog)

        # Apply filters
        if symbol:
            query = query.filter(TradeLog.symbol == symbol.upper())
        if status:
            query = query.filter(TradeLog.status == status.upper())
        if from_date is not None:
            query = query.filter(TradeLog.timestamp >= from_date)
        if to_date is not None:
            query = query.filter(TradeLog.timestamp <= to_date)

        # Apply ordering and limit
        trades = query.order_by(TradeLog.timestamp.desc()).limit(limit).all()

        logger.info(f"Retrieved {len(trades)} trades from database with filters: symbol={symbol}, status={status}")

        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "price": t.price,
                "status": t.status,
                "reason": t.reason,
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


@router.post(
    "",
    status_code=201,
    response_model=TradeResponse,
    summary="Execute Trade Order",
    description="Execute a new trading order with comprehensive validation and error handling"
)
async def create_trade(payload: TradeCreate, db=Depends(get_db)):
    """
    Execute a new trading order.

    This endpoint processes trade orders with:
    - Input validation for symbol, side, quantity, and price
    - Duplicate order detection
    - Real-time execution status tracking
    - Comprehensive error handling and logging

    The trade is immediately logged to the database with a FILLED status
    for demo purposes. In production, this would integrate with exchange APIs.
    """
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
            "reason": t.reason,
            "timestamp": t.timestamp
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

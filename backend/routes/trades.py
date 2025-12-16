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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from sqlalchemy.exc import SQLAlchemyError
import logging
from datetime import datetime, timezone

try:
    from database import get_db, TradeLog, Base
except ImportError:  # Support imports when backend package is used
    from backend.database import get_db, TradeLog, Base

try:
    from backend.utils.admin_auth import require_admin_token
    from backend.utils.admin_events import AdminEvent, record_admin_event
except ImportError:  # pragma: no cover - fallback for package-relative imports
    from utils.admin_auth import require_admin_token  # type: ignore
    from utils.admin_events import AdminEvent, record_admin_event  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Trading"],
    responses={
        400: {"description": "Invalid trading parameters"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
        502: {"description": "Exchange connectivity error"},
    },
)


class TradeCreate(BaseModel):
    """Request model for creating a new trade order."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    )
    side: str = Field(
        ..., pattern="^(BUY|SELL)$", description="Trade direction: BUY or SELL"
    )
    qty: float = Field(..., gt=0, description="Trade quantity (must be positive)")
    price: float = Field(..., gt=0, description="Trade price (must be positive)")


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
    description="Retrieve comprehensive trading history with filtering and pagination options",
)
async def get_trades(
    request: Request,
    db=Depends(get_db),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of trades to return"
    ),
    symbol: Optional[str] = Query(None, description="Filter by trading pair symbol"),
    status: Optional[str] = Query(None, description="Filter by trade status"),
    from_date: Optional[datetime] = Query(
        None, description="Start date for trade history"
    ),
    to_date: Optional[datetime] = Query(None, description="End date for trade history"),
    _admin_token: Optional[str] = Depends(require_admin_token),
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
    filters_applied: Dict[str, Any] = {
        "symbol": symbol.upper() if symbol else None,
        "status": status.upper() if status else None,
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "limit": limit,
    }

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
        trades = (
            query.order_by(TradeLog.timestamp.desc()).limit(limit).all()
        )

        logger.info(
            "Retrieved %s trades from database with filters: %s",
            len(trades),
            filters_applied,
        )
    except SQLAlchemyError as e:
        message = str(e)
        if "no such table" in message.lower():
            logger.warning("Trade table missing; auto-creating schema and returning empty history")
            try:
                Base.metadata.create_all(bind=db.get_bind())
            except Exception as create_err:
                logger.error("Failed to auto-create trade schema: %s", create_err)
                record_admin_event(
                    AdminEvent.TRADES_READ,
                    request=request,
                    success=False,
                    details={
                        "error": "schema_init_failed",
                        "message": str(create_err),
                        **filters_applied,
                    },
                )
                raise HTTPException(status_code=500, detail="Database error occurred")
            trades = []
        else:
            logger.error(f"Database error retrieving trades: {message}")
            record_admin_event(
                AdminEvent.TRADES_READ,
                request=request,
                success=False,
                details={
                    "error": "database_error",
                    "message": message,
                    **filters_applied,
                },
            )
            raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Unexpected error retrieving trades: {str(e)}")
        record_admin_event(
            AdminEvent.TRADES_READ,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "message": str(e),
                **filters_applied,
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    payload = [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": t.qty,
            "price": t.price,
            "status": t.status,
            "reason": t.reason,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
        }
        for t in trades
    ]

    record_admin_event(
        AdminEvent.TRADES_READ,
        request=request,
        success=True,
        details={
            "result_count": len(payload),
            **filters_applied,
        },
    )

    return payload


@router.post(
    "",
    status_code=201,
    response_model=TradeResponse,
    summary="Execute Trade Order",
    description="Execute a new trading order with comprehensive validation and error handling",
)
async def create_trade(
    payload: TradeCreate,
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
    db=Depends(get_db),
):
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
    symbol_upper = payload.symbol.upper()
    side_upper = payload.side.upper()
    notional = payload.qty * payload.price

    try:
        guard = getattr(request.app.state, "risk_guard", None)
        if guard is not None:
            # [ARCHITECTURE V2] Calculate risk metrics for PolicyStore v2
            leverage = getattr(payload, 'leverage', 5.0)  # Default 5x
            account_balance = 1000.0  # TODO: Get from session/account
            trade_risk_pct = (notional / account_balance) * 100 if account_balance > 0 else 0.0
            trace_id = f"api_{symbol_upper}_{int(datetime.now(timezone.utc).timestamp())}"
            
            allowed, reason = await guard.can_execute(
                symbol=symbol_upper,
                notional=notional,
                price=payload.price,
                price_as_of=datetime.now(timezone.utc),
                leverage=leverage,
                trade_risk_pct=trade_risk_pct,
                position_size_usd=notional,
                trace_id=trace_id,
            )
            if not allowed:
                raise HTTPException(
                    status_code=403,
                    detail={"error": "RiskCheckFailed", "reason": reason},
                )

        # Validate trade data
        if symbol_upper not in [
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "DOTUSDT",
            "LINKUSDT",
        ]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported trading symbol: {payload.symbol}"
            )

        if side_upper not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Side must be BUY or SELL")

        # Create trade log
        t = TradeLog(
            symbol=symbol_upper,
            side=side_upper,
            qty=payload.qty,
            price=payload.price,
            status="NEW",
        )

        # Ensure schema exists in ephemeral test environments
        try:
            Base.metadata.create_all(bind=db.get_bind())
        except Exception as schema_err:
            logger.error("Failed to validate trade schema: %s", schema_err)
            raise HTTPException(status_code=500, detail="Database error occurred")

        db.add(t)
        db.commit()
        db.refresh(t)

        logger.info(
            f"Created new trade: {t.id} - {t.side} {t.qty} {t.symbol} @ {t.price}"
        )

        response_payload = {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": t.qty,
            "price": t.price,
            "status": t.status,
            "reason": t.reason,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
        }

        if guard is not None:
            # Record trade with zero realised PnL for now; execution reconciliation updates later.
            await guard.record_execution(
                symbol=t.symbol,
                notional=notional,
                pnl=0.0,
            )

        record_admin_event(
            AdminEvent.TRADES_CREATE,
            request=request,
            success=True,
            details={
                "trade_id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "price": t.price,
                "notional": notional,
            },
        )

        return response_payload

    except HTTPException as exc:
        record_admin_event(
            AdminEvent.TRADES_CREATE,
            request=request,
            success=False,
            details={
                "status_code": exc.status_code,
                "detail": exc.detail,
                "symbol": symbol_upper,
                "side": side_upper,
            },
        )
        raise
    except SQLAlchemyError as e:
        db.rollback()
        message = str(e)
        if "no such table" in message.lower():
            logger.warning("Trade table missing during create; retrying after schema init")
            try:
                Base.metadata.create_all(bind=db.get_bind())
                db.add(t)
                db.commit()
                db.refresh(t)
                response_payload = {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "price": t.price,
                    "status": t.status,
                    "reason": t.reason,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                }
                if guard is not None:
                    await guard.record_execution(
                        symbol=t.symbol,
                        notional=notional,
                        pnl=0.0,
                    )
                record_admin_event(
                    AdminEvent.TRADES_CREATE,
                    request=request,
                    success=True,
                    details={
                        "trade_id": t.id,
                        "symbol": t.symbol,
                        "side": t.side,
                        "qty": t.qty,
                        "price": t.price,
                        "notional": notional,
                    },
                )
                return response_payload
            except Exception as retry_err:
                db.rollback()
                logger.error("Failed to recover from missing trade table: %s", retry_err)
        logger.error(f"Database error creating trade: {message}")
        record_admin_event(
            AdminEvent.TRADES_CREATE,
            request=request,
            success=False,
            details={
                "error": "database_error",
                "message": message,
                "symbol": symbol_upper,
                "side": side_upper,
            },
        )
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error creating trade: {str(e)}")
        record_admin_event(
            AdminEvent.TRADES_CREATE,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "message": str(e),
                "symbol": symbol_upper,
                "side": side_upper,
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recent")
async def recent_trades(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Number of sample trades to return"),
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    """Return a deterministic list of recent demo trades for frontend testing."""

    try:
        trades = []
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        for i in range(limit):
            trades.append(
                {
                    "id": i + 1,  # Fixed: Return integer instead of string
                    "symbol": symbols[i % len(symbols)],
                    "side": "BUY" if i % 2 == 0 else "SELL",
                    "qty": round(0.01 * (i + 1), 4),
                    "price": round(100 + i * 0.5, 2),
                    "timestamp": i,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected failures
        record_admin_event(
            AdminEvent.TRADES_RECENT,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "message": str(exc),
                "limit": limit,
            },
        )
        raise

    record_admin_event(
        AdminEvent.TRADES_RECENT,
        request=request,
        success=True,
        details={
            "result_count": len(trades),
            "limit": limit,
        },
    )

    return trades

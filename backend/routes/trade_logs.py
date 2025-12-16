# backend/routes/trade_logs.py
from typing import Annotated, Optional

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.exc import SQLAlchemyError

try:
    from backend.database import get_db, TradeLog, Base
except ImportError:  # pragma: no cover - fallback for legacy flat layout
    from database import get_db, TradeLog, Base  # type: ignore

try:
    from backend.utils.admin_auth import require_admin_token
    from backend.utils.admin_events import AdminEvent, record_admin_event
except ImportError:  # pragma: no cover - fallback for package layout
    from utils.admin_auth import require_admin_token  # type: ignore
    from utils.admin_events import AdminEvent, record_admin_event  # type: ignore


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/trade_logs")
async def get_trade_logs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    db=Depends(get_db),
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    """
    Returner de siste trade-loggene fra databasen med administrativ tilgang.
    :param limit: Hvor mange logs som skal returneres (default 50, max 500)
    """

    try:
        rows = (
            db.query(TradeLog)
            .order_by(TradeLog.id.desc())
            .limit(limit)
            .with_entities(
                TradeLog.timestamp,
                TradeLog.symbol,
                TradeLog.side,
                TradeLog.qty,
                TradeLog.price,
                TradeLog.status,
                TradeLog.reason,
            )
            .all()
        )
    except SQLAlchemyError as exc:
        message = str(exc)
        if "no such table" in message.lower():
            logger.warning("Trade logs-tabellen mangler; oppretter schema og returnerer tomt resultat")
            try:
                Base.metadata.create_all(bind=db.get_bind())
            except Exception as schema_err:  # pragma: no cover - defensive fallback
                logger.error("Feil ved oppretting av trade logs-tabell: %s", schema_err)
                record_admin_event(
                    AdminEvent.TRADE_LOGS_READ,
                    request=request,
                    success=False,
                    details={
                        "error": "schema_init_failed",
                        "message": str(schema_err),
                        "limit": limit,
                    },
                )
                raise HTTPException(status_code=500, detail="Database error occurred") from schema_err
            rows = []
        else:
            logger.error("Databasefeil ved henting av trade logs: %s", message)
            record_admin_event(
                AdminEvent.TRADE_LOGS_READ,
                request=request,
                success=False,
                details={
                    "error": "database_error",
                    "message": message,
                    "limit": limit,
                },
            )
            raise HTTPException(status_code=500, detail="Database error occurred") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Uventet feil ved henting av trade logs")
        record_admin_event(
            AdminEvent.TRADE_LOGS_READ,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "message": str(exc),
                "limit": limit,
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    logs = [
        {
            "timestamp": ts.isoformat() if ts else None,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "status": status,
            "reason": reason,
        }
        for ts, symbol, side, qty, price, status, reason in rows
    ]

    record_admin_event(
        AdminEvent.TRADE_LOGS_READ,
        request=request,
        success=True,
        details={
            "result_count": len(logs),
            "limit": limit,
        },
    )

    return {"logs": logs}

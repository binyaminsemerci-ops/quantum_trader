"""Administrative liquidity endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.config.liquidity import load_liquidity_config
from backend.database import get_db
from backend.services.liquidity import refresh_liquidity
from backend.utils.admin_auth import require_admin_token
from backend.utils.admin_events import AdminEvent, record_admin_event


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/liquidity",
    tags=["Liquidity"],
    responses={
        401: {"description": "Missing admin token"},
        403: {"description": "Invalid admin token"},
        502: {"description": "Failed to refresh liquidity"},
    },
)


@router.post(
    "/refresh",
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_liquidity_refresh(
    request: Request,
    db: Session = Depends(get_db),
    _admin_token: Optional[str] = Depends(require_admin_token),
) -> Dict[str, Any]:
    """Refresh the liquidity universe using configured providers."""

    config = load_liquidity_config()
    try:
        result = await refresh_liquidity(db, config=config)
        record_admin_event(
            AdminEvent.LIQUIDITY_REFRESH,
            request=request,
            details=result,
        )
        payload = {"status": "accepted", **result}
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.exception("Liquidity refresh failed", exc_info=exc)
        record_admin_event(
            AdminEvent.LIQUIDITY_REFRESH,
            request=request,
            success=False,
            details={"error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to refresh liquidity universe",
        ) from exc


__all__ = ["router"]

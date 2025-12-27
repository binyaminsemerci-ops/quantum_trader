"""Administrative scheduler control endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

try:  # pragma: no cover - support package-relative imports
    from backend.utils.admin_auth import require_admin_token
    from backend.utils.admin_events import AdminEvent, record_admin_event
    from backend.utils.scheduler import (
        get_configured_symbols,
        get_scheduler_snapshot,
        run_execution_cycle_now,
        run_liquidity_refresh_now,
        run_market_cache_refresh_now,
    )
except ImportError:  # pragma: no cover
    from utils.admin_auth import require_admin_token  # type: ignore
    from utils.admin_events import AdminEvent, record_admin_event  # type: ignore
    from utils.scheduler import (  # type: ignore
        get_configured_symbols,
        get_scheduler_snapshot,
        run_execution_cycle_now,
        run_liquidity_refresh_now,
        run_market_cache_refresh_now,
    )


logger = logging.getLogger(__name__)


class WarmRequest(BaseModel):
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Optional list of symbols to refresh; defaults to the configured scheduler universe.",
    )


router = APIRouter(
    prefix="/scheduler",
    tags=["Scheduler"],
    responses={
        401: {"description": "Missing admin token"},
        403: {"description": "Invalid admin token"},
        503: {"description": "Scheduler operation failed"},
    },
)


@router.get("/status")
async def get_scheduler_status(
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
) -> Dict[str, Any]:
    """Return a snapshot of scheduler configuration and recent runs."""

    snapshot = get_scheduler_snapshot()
    record_admin_event(
        AdminEvent.SCHEDULER_STATUS,
        request=request,
        details={
            "market_status": snapshot.get("last_run", {}).get("status"),
            "liquidity_status": snapshot.get("liquidity", {}).get("status"),
            "execution_status": snapshot.get("execution", {}).get("status"),
        },
    )
    return snapshot


@router.post("/warm", status_code=status.HTTP_202_ACCEPTED)
async def trigger_market_warm(
    payload: WarmRequest,
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
) -> Dict[str, Any]:
    """Trigger a one-off market cache refresh for the scheduler symbol set."""

    symbols = payload.symbols or get_configured_symbols()
    try:
        result = await run_market_cache_refresh_now(symbols)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual market cache warm-up failed", exc_info=exc)
        record_admin_event(
            AdminEvent.SCHEDULER_WARM,
            request=request,
            success=False,
            details={"symbols": symbols, "error": str(exc)},
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Market cache refresh failed") from exc

    record_admin_event(
        AdminEvent.SCHEDULER_WARM,
        request=request,
        details={"symbols": symbols, "status": result.get("last_run", {}).get("status")},
    )
    return {"status": "accepted", **result}


@router.post("/liquidity", status_code=status.HTTP_202_ACCEPTED)
async def trigger_liquidity_refresh(
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
) -> Dict[str, Any]:
    """Run the liquidity refresh job immediately."""

    try:
        state = await run_liquidity_refresh_now()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual liquidity refresh failed", exc_info=exc)
        record_admin_event(
            AdminEvent.SCHEDULER_LIQUIDITY,
            request=request,
            success=False,
            details={"error": str(exc)},
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Liquidity refresh failed") from exc

    record_admin_event(
        AdminEvent.SCHEDULER_LIQUIDITY,
        request=request,
        details={
            "status": state.get("status"),
            "run_id": state.get("run_id"),
            "selection_size": state.get("selection_size"),
        },
    )
    return {"status": "accepted", "liquidity": state}


@router.post("/execution", status_code=status.HTTP_202_ACCEPTED)
async def trigger_execution_cycle(
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
) -> Dict[str, Any]:
    """Run the execution rebalance cycle immediately."""

    try:
        state = await run_execution_cycle_now()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual execution cycle failed", exc_info=exc)
        record_admin_event(
            AdminEvent.SCHEDULER_EXECUTION,
            request=request,
            success=False,
            details={"error": str(exc)},
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Execution cycle failed") from exc

    record_admin_event(
        AdminEvent.SCHEDULER_EXECUTION,
        request=request,
        details={
            "status": state.get("status"),
            "orders_planned": state.get("orders_planned"),
            "orders_submitted": state.get("orders_submitted"),
        },
    )
    return {"status": "accepted", "execution": state}


__all__ = ["router"]

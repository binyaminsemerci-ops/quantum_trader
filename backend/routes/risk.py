"""Risk management administrative API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

try:  # Support running from backend package root and top-level imports
    from services.risk_guard import RiskGuardService  # type: ignore
    from utils.admin_auth import enforce_admin_token  # type: ignore
    from utils.admin_events import AdminEvent, record_admin_event  # type: ignore
except ImportError:  # pragma: no cover - fallback when imported as package
    from backend.services.risk.risk_guard import RiskGuardService
    from backend.utils.admin_auth import enforce_admin_token
    from backend.utils.admin_events import AdminEvent, record_admin_event


router = APIRouter(
    prefix="/risk",
    tags=["Risk"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Access forbidden"},
        404: {"description": "Resource not found"},
        503: {"description": "Risk guard unavailable"},
    },
)


class RiskRecord(BaseModel):
    timestamp: datetime
    notional: float
    pnl: float


class RiskState(BaseModel):
    kill_switch_override: Optional[bool]
    kill_switch_state: Optional["KillSwitchStateModel"] = None
    records: List[RiskRecord]
    daily_loss: float
    trade_count: int


class RiskConfigModel(BaseModel):
    staging_mode: bool
    kill_switch: bool
    max_notional_per_trade: float
    max_daily_loss: float
    allowed_symbols: List[str]
    failsafe_reset_minutes: int
    risk_state_db_path: Optional[str]
    max_position_per_symbol: Optional[float]
    max_gross_exposure: Optional[float]


class PositionEntry(BaseModel):
    symbol: str
    quantity: float
    notional: float
    updated_at: Optional[datetime] = None


class PositionsSnapshot(BaseModel):
    positions: List[PositionEntry] = Field(default_factory=list)
    total_notional: float = 0.0
    as_of: Optional[datetime] = None


class RiskSnapshot(BaseModel):
    config: RiskConfigModel
    state: RiskState
    positions: PositionsSnapshot


class KillSwitchUpdate(BaseModel):
    enabled: Optional[bool] = Field(
        None,
        description="Set to true or false to override configuration, null to revert to defaults.",
    )
    reason: Optional[str] = Field(
        None,
        description="Operator-supplied reason for engaging or clearing the kill switch.",
        max_length=256,
    )


class ResetResponse(BaseModel):
    status: str = Field("reset", description="Status indicator for risk guard reset")
    snapshot: RiskSnapshot


class KillSwitchResponse(BaseModel):
    status: str = Field("success", description="Status indicator for kill switch update")
    snapshot: RiskSnapshot


class KillSwitchStateModel(BaseModel):
    enabled: bool
    reason: str
    updated_at: datetime


RiskState.model_rebuild()


async def _get_risk_guard(request: Request) -> RiskGuardService:
    guard = getattr(request.app.state, "risk_guard", None)
    if guard is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk guard service not initialised",
        )
    return guard


async def _require_admin_access(
    request: Request,
    admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> RiskGuardService:
    guard = await _get_risk_guard(request)
    enforce_admin_token(
        request=request,
        provided_token=admin_token,
        expected_token=guard.config.admin_api_token,
        missing_event=AdminEvent.RISK_AUTH_MISSING,
        invalid_event=AdminEvent.RISK_AUTH_INVALID,
    )
    return guard


@router.get("", response_model=RiskSnapshot, summary="Get risk guard snapshot")
async def get_risk_snapshot(
    request: Request,
    risk_guard: RiskGuardService = Depends(_require_admin_access),
) -> RiskSnapshot:
    snapshot = await risk_guard.snapshot()
    record_admin_event(
        AdminEvent.RISK_SNAPSHOT,
        request=request,
        success=True,
        details={
            "trade_count": snapshot["state"]["trade_count"],
            "kill_switch_override": snapshot["state"].get("kill_switch_override"),
        },
    )
    return RiskSnapshot(**snapshot)


@router.post(
    "/kill-switch",
    response_model=KillSwitchResponse,
    summary="Override kill switch state",
    description="Toggle the risk guard kill switch override or reset to configuration defaults.",
)
async def set_kill_switch(
    payload: KillSwitchUpdate,
    request: Request,
    risk_guard: RiskGuardService = Depends(_require_admin_access),
) -> KillSwitchResponse:
    await risk_guard.set_kill_switch(payload.enabled, reason=payload.reason)
    snapshot = await risk_guard.snapshot()
    record_admin_event(
        AdminEvent.RISK_KILL_SWITCH,
        request=request,
        success=True,
        details={
            "override": payload.enabled,
            "reason": payload.reason,
        },
    )
    return KillSwitchResponse(snapshot=RiskSnapshot(**snapshot))


@router.get(
    "/kill-switch",
    response_model=KillSwitchResponse,
    summary="Get kill switch state",
    description="Retrieve the latest kill switch snapshot without mutating state.",
)
async def get_kill_switch_state(
    request: Request,
    risk_guard: RiskGuardService = Depends(_require_admin_access),
) -> KillSwitchResponse:
    snapshot = await risk_guard.snapshot()
    state = snapshot.get("state", {}) if isinstance(snapshot, dict) else {}
    kill_switch_state = state.get("kill_switch_state") if isinstance(state, dict) else None
    record_admin_event(
        AdminEvent.RISK_KILL_SWITCH,
        request=request,
        success=True,
        details={
            "override": state.get("kill_switch_override") if isinstance(state, dict) else None,
            "reason": kill_switch_state.get("reason") if isinstance(kill_switch_state, dict) else None,
            "method": "GET",
        },
    )
    return KillSwitchResponse(snapshot=RiskSnapshot(**snapshot))


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Reset risk guard state",
    description="Clear trade history and kill switch overrides maintained by the risk guard.",
)
async def reset_risk_guard(
    request: Request,
    risk_guard: RiskGuardService = Depends(_require_admin_access),
) -> ResetResponse:
    await risk_guard.reset()
    snapshot = await risk_guard.snapshot()
    record_admin_event(
        AdminEvent.RISK_RESET,
        request=request,
        success=True,
        details={"trade_count": snapshot["state"]["trade_count"]},
    )
    return ResetResponse(snapshot=RiskSnapshot(**snapshot))

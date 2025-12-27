"""
Federation AI FastAPI App
==========================

RESTful API for Federation AI service.

Endpoints:
- GET /health - Health check
- GET /api/federation/status - Get status of all roles
- GET /api/federation/decisions - Get recent decisions
- POST /api/federation/mode - Manually override trading mode
- POST /api/federation/risk - Manually adjust risk limits
- POST /api/federation/roles/{role_name}/enable - Enable role
- POST /api/federation/roles/{role_name}/disable - Disable role
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import structlog

from backend.services.federation_ai.orchestrator import FederationOrchestrator
from backend.services.federation_ai.models import (
    FederationDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    ModelPerformance,
)

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Federation AI v3",
    description="Executive AI orchestration service",
    version="3.0.0"
)

# Global orchestrator instance
orchestrator: Optional[FederationOrchestrator] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    roles_active: int


class StatusResponse(BaseModel):
    roles: Dict[str, bool]
    total_decisions: int
    last_update: Optional[str]


class ManualModeRequest(BaseModel):
    mode: str  # LIVE, SHADOW, PAUSED, EMERGENCY
    reason: str
    duration_minutes: Optional[int] = None


class ManualRiskRequest(BaseModel):
    max_leverage: Optional[float] = None
    max_position_size_usd: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    reason: str


@app.on_event("startup")
async def startup_event():
    """Initialize Federation orchestrator on startup"""
    global orchestrator
    orchestrator = FederationOrchestrator()
    logger.info("Federation AI service started")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    role_status = orchestrator.get_role_status()
    active_roles = sum(1 for enabled in role_status.values() if enabled)
    
    return HealthResponse(
        status="healthy",
        service="federation-ai",
        version="3.0.0",
        roles_active=active_roles,
    )


@app.get("/api/federation/status", response_model=StatusResponse)
async def get_status():
    """Get status of all Federation roles"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    role_status = orchestrator.get_role_status()
    decision_history = orchestrator.get_decision_history(limit=1)
    
    return StatusResponse(
        roles=role_status,
        total_decisions=len(orchestrator.decision_history),
        last_update=decision_history[0].timestamp if decision_history else None,
    )


@app.get("/api/federation/decisions")
async def get_decisions(limit: int = 50):
    """Get recent Federation decisions"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    decisions = orchestrator.get_decision_history(limit=limit)
    
    return {
        "total": len(decisions),
        "decisions": [d.model_dump() for d in decisions],
    }


@app.post("/api/federation/mode")
async def set_trading_mode(request: ManualModeRequest):
    """
    Manually override trading mode.
    
    Use case: Emergency shutdown, testing, maintenance
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Validate mode
    valid_modes = ["LIVE", "SHADOW", "PAUSED", "EMERGENCY"]
    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )
    
    # Create manual decision
    from backend.services.federation_ai.models import DecisionType, DecisionPriority
    
    manual_decision = FederationDecision(
        decision_type=DecisionType.MODE_CHANGE,
        role_source="manual",
        priority=DecisionPriority.CRITICAL,
        reason=f"Manual override: {request.reason}",
        payload={
            "mode": request.mode,
            "duration_minutes": request.duration_minutes,
        },
    )
    
    # Publish decision
    await orchestrator._publish_decision(manual_decision)
    
    logger.warning(
        "Manual mode override",
        mode=request.mode,
        reason=request.reason,
    )
    
    return {
        "success": True,
        "decision_id": manual_decision.decision_id,
        "mode": request.mode,
    }


@app.post("/api/federation/risk")
async def adjust_risk_limits(request: ManualRiskRequest):
    """
    Manually adjust risk limits.
    
    Use case: Testing, emergency tightening
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Build risk adjustment payload
    risk_payload = {}
    if request.max_leverage is not None:
        risk_payload["max_leverage"] = request.max_leverage
    if request.max_position_size_usd is not None:
        risk_payload["max_position_size_usd"] = request.max_position_size_usd
    if request.max_drawdown_pct is not None:
        risk_payload["max_drawdown_pct"] = request.max_drawdown_pct
    
    if not risk_payload:
        raise HTTPException(
            status_code=400,
            detail="At least one risk parameter must be provided"
        )
    
    # Create manual decision
    from backend.services.federation_ai.models import DecisionType, DecisionPriority
    
    manual_decision = FederationDecision(
        decision_type=DecisionType.RISK_ADJUSTMENT,
        role_source="manual",
        priority=DecisionPriority.CRITICAL,
        reason=f"Manual risk adjustment: {request.reason}",
        payload=risk_payload,
    )
    
    # Publish decision
    await orchestrator._publish_decision(manual_decision)
    
    logger.warning(
        "Manual risk adjustment",
        limits=risk_payload,
        reason=request.reason,
    )
    
    return {
        "success": True,
        "decision_id": manual_decision.decision_id,
        "limits": risk_payload,
    }


@app.post("/api/federation/roles/{role_name}/enable")
async def enable_role(role_name: str):
    """Enable a specific Federation role"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    valid_roles = ["ceo", "cio", "cro", "cfo", "researcher", "supervisor"]
    if role_name not in valid_roles:
        raise HTTPException(
            status_code=404,
            detail=f"Role not found. Valid roles: {valid_roles}"
        )
    
    orchestrator.enable_role(role_name)
    
    return {
        "success": True,
        "role": role_name,
        "status": "enabled",
    }


@app.post("/api/federation/roles/{role_name}/disable")
async def disable_role(role_name: str):
    """Disable a specific Federation role"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    valid_roles = ["ceo", "cio", "cro", "cfo", "researcher", "supervisor"]
    if role_name not in valid_roles:
        raise HTTPException(
            status_code=404,
            detail=f"Role not found. Valid roles: {valid_roles}"
        )
    
    # Don't allow disabling supervisor (safety)
    if role_name == "supervisor":
        raise HTTPException(
            status_code=403,
            detail="Cannot disable supervisor role (safety constraint)"
        )
    
    orchestrator.disable_role(role_name)
    
    return {
        "success": True,
        "role": role_name,
        "status": "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

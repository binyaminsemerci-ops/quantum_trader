"""
PolicyStore API Routes

Exposes HTTP endpoints for viewing and updating the global trading policy.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/policy", tags=["policy"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PolicyResponse(BaseModel):
    """Response model for policy endpoints."""
    policy: Dict[str, Any]
    timestamp: str


class PolicyUpdateRequest(BaseModel):
    """Request model for partial policy updates."""
    risk_mode: str | None = Field(None, pattern="^(AGGRESSIVE|NORMAL|DEFENSIVE)$")
    max_risk_per_trade: float | None = Field(None, ge=0, le=1)
    max_positions: int | None = Field(None, ge=1, le=100)
    global_min_confidence: float | None = Field(None, ge=0, le=1)
    allowed_strategies: list[str] | None = None
    allowed_symbols: list[str] | None = None


class PolicyStatusResponse(BaseModel):
    """Response model for policy status."""
    available: bool
    current_risk_mode: str | None
    last_updated: str | None
    timestamp: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_policy_store(request: Request):
    """Get PolicyStore from app state."""
    policy_store = getattr(request.app.state, 'policy_store', None)
    if policy_store is None:
        raise HTTPException(
            status_code=503,
            detail="PolicyStore not initialized"
        )
    return policy_store


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/status", response_model=PolicyStatusResponse)
async def get_policy_status(request: Request):
    """
    Get PolicyStore availability and current status.
    
    Returns basic info about the policy system without full details.
    """
    try:
        policy_store = _get_policy_store(request)
        policy = policy_store.get()
        
        return PolicyStatusResponse(
            available=True,
            current_risk_mode=policy.get('risk_mode'),
            last_updated=policy.get('last_updated'),
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        return PolicyStatusResponse(
            available=False,
            current_risk_mode=None,
            last_updated=None,
            timestamp=datetime.now().isoformat()
        )


@router.get("", response_model=PolicyResponse)
@router.get("/", response_model=PolicyResponse)
async def get_current_policy(request: Request):
    """
    Get the current global trading policy.
    
    Returns all policy fields including:
    - risk_mode
    - allowed_strategies
    - allowed_symbols
    - max_risk_per_trade
    - max_positions
    - global_min_confidence
    - opp_rankings
    - model_versions
    """
    policy_store = _get_policy_store(request)
    policy = policy_store.get()
    
    return PolicyResponse(
        policy=policy,
        timestamp=datetime.now().isoformat()
    )


@router.patch("", response_model=PolicyResponse)
@router.patch("/", response_model=PolicyResponse)
async def update_policy(request: Request, update: PolicyUpdateRequest):
    """
    Update specific policy fields.
    
    Only provided fields will be updated, others remain unchanged.
    Validates all inputs before applying changes.
    
    Example:
        PATCH /api/policy
        {
            "risk_mode": "AGGRESSIVE",
            "max_risk_per_trade": 0.015
        }
    """
    policy_store = _get_policy_store(request)
    
    # Build update dict from non-None fields
    update_dict = {}
    if update.risk_mode is not None:
        update_dict['risk_mode'] = update.risk_mode
    if update.max_risk_per_trade is not None:
        update_dict['max_risk_per_trade'] = update.max_risk_per_trade
    if update.max_positions is not None:
        update_dict['max_positions'] = update.max_positions
    if update.global_min_confidence is not None:
        update_dict['global_min_confidence'] = update.global_min_confidence
    if update.allowed_strategies is not None:
        update_dict['allowed_strategies'] = update.allowed_strategies
    if update.allowed_symbols is not None:
        update_dict['allowed_symbols'] = update.allowed_symbols
    
    if not update_dict:
        raise HTTPException(
            status_code=400,
            detail="No fields provided for update"
        )
    
    # Apply update
    try:
        policy_store.patch(update_dict)
        logger.info(f"[PolicyStore API] Updated policy: {update_dict}")
    except Exception as e:
        logger.error(f"[PolicyStore API] Update failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Policy update failed: {str(e)}"
        )
    
    # Return updated policy
    policy = policy_store.get()
    return PolicyResponse(
        policy=policy,
        timestamp=datetime.now().isoformat()
    )


@router.post("/reset", response_model=PolicyResponse)
async def reset_policy(request: Request):
    """
    Reset policy to default values.
    
    WARNING: This will discard all current settings and restore defaults.
    """
    policy_store = _get_policy_store(request)
    
    try:
        policy_store.reset()
        logger.warning("[PolicyStore API] Policy reset to defaults")
    except Exception as e:
        logger.error(f"[PolicyStore API] Reset failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Policy reset failed: {str(e)}"
        )
    
    # Return new policy
    policy = policy_store.get()
    return PolicyResponse(
        policy=policy,
        timestamp=datetime.now().isoformat()
    )


@router.get("/risk_mode", response_model=Dict[str, str])
async def get_risk_mode(request: Request):
    """
    Get current risk mode only.
    
    Lightweight endpoint for quick risk mode checks.
    """
    policy_store = _get_policy_store(request)
    policy = policy_store.get()
    
    return {
        "risk_mode": policy.get('risk_mode', 'UNKNOWN'),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/risk_mode/{mode}")
async def set_risk_mode(request: Request, mode: str):
    """
    Set risk mode directly.
    
    Args:
        mode: One of AGGRESSIVE, NORMAL, DEFENSIVE
    
    Example:
        POST /api/policy/risk_mode/AGGRESSIVE
    """
    mode = mode.upper()
    if mode not in ["AGGRESSIVE", "NORMAL", "DEFENSIVE"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk mode: {mode}. Must be AGGRESSIVE, NORMAL, or DEFENSIVE"
        )
    
    policy_store = _get_policy_store(request)
    
    try:
        policy_store.patch({"risk_mode": mode})
        logger.info(f"[PolicyStore API] Risk mode changed to {mode}")
    except Exception as e:
        logger.error(f"[PolicyStore API] Risk mode change failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk mode update failed: {str(e)}"
        )
    
    policy = policy_store.get()
    return {
        "risk_mode": policy['risk_mode'],
        "max_risk_per_trade": policy['max_risk_per_trade'],
        "max_positions": policy['max_positions'],
        "global_min_confidence": policy['global_min_confidence'],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/allowed_symbols")
async def get_allowed_symbols(request: Request):
    """Get list of currently allowed trading symbols."""
    policy_store = _get_policy_store(request)
    policy = policy_store.get()
    
    return {
        "allowed_symbols": policy.get('allowed_symbols', []),
        "count": len(policy.get('allowed_symbols', [])),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/model_versions")
async def get_model_versions(request: Request):
    """Get active ML model versions."""
    policy_store = _get_policy_store(request)
    policy = policy_store.get()
    
    return {
        "model_versions": policy.get('model_versions', {}),
        "timestamp": datetime.now().isoformat()
    }

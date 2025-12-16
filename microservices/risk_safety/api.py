"""
Risk & Safety Service - REST API Routes
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ESSOverrideRequest(BaseModel):
    """Request to override ESS state."""
    new_state: str = Field(..., description="Target ESS state (NORMAL, CAUTION, CRITICAL)")
    reason: str = Field(..., description="Reason for override")


class PolicyUpdateRequest(BaseModel):
    """Request to update policy value."""
    key: str = Field(..., description="Policy key")
    value: Any = Field(..., description="Policy value")


# ============================================================================
# ESS ENDPOINTS
# ============================================================================

@router.get("/risk/ess/status")
async def get_ess_status():
    """
    Get current ESS (Emergency Stop System) status.
    
    Returns:
        - state: Current ESS state (NORMAL, CAUTION, CRITICAL)
        - can_execute: Whether orders can be placed
        - trip_reason: Reason for CRITICAL state (if applicable)
        - metrics: ESS tracking metrics
    """
    from .main import service
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return await service.get_ess_status()


@router.post("/risk/ess/override")
async def override_ess(request: ESSOverrideRequest):
    """
    Manually override ESS state.
    
    WARNING: Use with caution. This bypasses ESS safety logic.
    """
    from .main import service
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Validate state
    valid_states = ["NORMAL", "CAUTION", "CRITICAL"]
    if request.new_state not in valid_states:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state. Must be one of: {valid_states}"
        )
    
    result = await service.override_ess(request.new_state, request.reason)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


@router.post("/risk/ess/reset")
async def reset_ess():
    """
    Reset ESS to NORMAL state.
    
    Clears all ESS metrics and transitions to NORMAL.
    """
    from .main import service
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    result = await service.reset_ess()
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


# ============================================================================
# POLICY ENDPOINTS
# ============================================================================

@router.get("/policy/{key}")
async def get_policy(key: str):
    """
    Get policy value by key.
    
    Args:
        key: Policy key (e.g., "execution.max_slippage_pct")
    
    Returns:
        Policy value or null if not found
    """
    from .main import service
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    value = await service.get_policy(key)
    
    return {
        "key": key,
        "value": value
    }


@router.get("/policy")
async def get_all_policies():
    """
    Get all policy values.
    
    Returns:
        Dictionary of all policies
    """
    from .main import service
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    policies = await service.get_all_policies()
    
    return {
        "policies": policies,
        "count": len(policies)
    }


@router.post("/policy/update")
async def update_policy(request: PolicyUpdateRequest):
    """
    Update policy value.
    
    This will trigger a "policy.updated" event that other services can consume.
    """
    from .main import service
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    result = await service.update_policy(request.key, request.value)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


# ============================================================================
# RISK LIMIT ENDPOINTS (Future)
# ============================================================================

@router.get("/risk/limits/{symbol}")
async def get_risk_limits(symbol: str):
    """
    Get risk limits for a symbol.
    
    TODO: Implement risk limit calculation based on:
    - Symbol volatility
    - Portfolio exposure
    - ESS state
    - PolicyStore configuration
    """
    # Placeholder
    return {
        "symbol": symbol,
        "max_position_size_usd": 1000.0,
        "max_leverage": 10,
        "max_exposure_pct": 0.20,
        "status": "not_implemented"
    }

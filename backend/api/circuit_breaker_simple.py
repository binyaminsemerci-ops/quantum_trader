"""
Circuit Breaker Management API - Simple Version
===============================================
Manages the global circuit breaker state in Auto Executor

Endpoints:
- GET  /api/circuit-breaker/status   - Current state
- POST /api/circuit-breaker/reset    - Reset with safety checks
- GET  /api/circuit-breaker/history  - Event history

Created: 2024-12-21 (Phase 2)
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger("circuit_breaker_api")

# Import Auto Executor module to access global variables
import backend.microservices.auto_executor.executor_service as auto_executor


class StatusResponse(BaseModel):
    """Circuit breaker status"""
    active: bool
    until: Optional[str]  # ISO timestamp
    time_remaining_seconds: Optional[float]
    can_reset: bool
    blocked_orders: int
    total_activations: int


class ResetRequest(BaseModel):
    """Reset request"""
    reason: str
    override: bool = False


class HistoryEntry(BaseModel):
    """History entry"""
    timestamp: str
    event: str
    reason: Optional[str] = None


# Global tracking
_history: List[dict] = []
_blocked_orders = 0
_total_activations = 0


def register_routes(app):
    """Register circuit breaker routes"""
    
    @app.get("/api/circuit-breaker/status", tags=["Circuit Breaker"])
    async def get_status() -> StatusResponse:
        """Get circuit breaker status"""
        global _blocked_orders, _total_activations
        
        active = auto_executor.circuit_breaker_active
        until_ts = auto_executor.circuit_breaker_until
        
        # Calculate time remaining
        until = None
        time_remaining = None
        if active and until_ts > 0:
            until = datetime.fromtimestamp(until_ts).isoformat()
            time_remaining = until_ts - datetime.now().timestamp()
            if time_remaining < 0:
                time_remaining = 0
        
        # Safe to reset if more than 1 hour has passed
        can_reset = True
        if active and time_remaining and time_remaining > 3600:
            can_reset = False
        
        return StatusResponse(
            active=active,
            until=until,
            time_remaining_seconds=time_remaining,
            can_reset=can_reset,
            blocked_orders=_blocked_orders,
            total_activations=_total_activations
        )
    
    @app.post("/api/circuit-breaker/reset", tags=["Circuit Breaker"])
    async def reset_breaker(request: ResetRequest) -> dict:
        """Reset circuit breaker"""
        global _history
        
        active = auto_executor.circuit_breaker_active
        
        if not active:
            return {"message": "Circuit breaker not active", "state": "already_clear"}
        
        # Safety check
        until_ts = auto_executor.circuit_breaker_until
        time_remaining = until_ts - datetime.now().timestamp()
        
        if time_remaining > 3600 and not request.override:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot reset: {time_remaining/3600:.1f}h remaining. Use override=true."
            )
        
        # Reset
        auto_executor.circuit_breaker_active = False
        auto_executor.circuit_breaker_until = 0
        
        # Log
        _history.append({
            "timestamp": datetime.now().isoformat(),
            "event": "reset" if not request.override else "reset_override",
            "reason": request.reason
        })
        
        logger.warning(f"[CB] Reset: {request.reason} (override={request.override})")
        
        return {
            "message": "Circuit breaker reset",
            "reason": request.reason,
            "override": request.override
        }
    
    @app.get("/api/circuit-breaker/history", tags=["Circuit Breaker"])
    async def get_history(limit: int = 50) -> List[dict]:
        """Get history"""
        return list(reversed(_history[-limit:]))
    
    logger.info("✅ Circuit Breaker API registered")


def record_activation(reason: str = "drawdown_exceeded"):
    """Record activation (call from Auto Executor)"""
    global _history, _total_activations
    
    _total_activations += 1
    _history.append({
        "timestamp": datetime.now().isoformat(),
        "event": "activated",
        "reason": reason
    })


def record_blocked_order():
    """Record blocked order (call from Auto Executor)"""
    global _blocked_orders
    _blocked_orders += 1

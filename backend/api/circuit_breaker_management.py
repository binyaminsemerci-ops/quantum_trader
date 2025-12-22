"""
Circuit Breaker Management API
===============================
Add management endpoints for circuit breaker observability and control
"""
from fastapi import HTTPException
from typing import Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def register_circuit_breaker_routes(app, circuit_breaker):
    """Register circuit breaker management routes"""
    
    @app.get("/api/circuit-breaker/status")
    async def get_circuit_breaker_status():
        """
        Get current circuit breaker status
        
        Returns detailed information about:
        - Current state (OPEN/CLOSED/HALF_OPEN)
        - Failure count
        - Last failure timestamp and reason
        - Whether it can be safely reset
        - Statistics on blocked vs allowed orders
        """
        try:
            # Get circuit breaker instance
            cb = circuit_breaker
            
            # Get current state
            state = getattr(cb, 'state', 'UNKNOWN')
            is_active = getattr(cb, 'is_active', lambda: False)()
            failure_count = getattr(cb, 'failure_count', 0)
            last_failure = getattr(cb, 'last_failure_time', None)
            trigger_reason = getattr(cb, 'trigger_reason', None)
            threshold = getattr(cb, 'threshold', None)
            
            # Check if safe to reset
            can_reset = False
            reset_reason = "Unknown"
            
            if hasattr(cb, 'can_reset'):
                can_reset = cb.can_reset()
                if not can_reset:
                    # Get reason why we can't reset
                    if failure_count >= threshold:
                        reset_reason = f"Failure count ({failure_count}) still above threshold ({threshold})"
                    elif last_failure and (datetime.now(timezone.utc) - last_failure).seconds < 300:
                        reset_reason = "Last failure too recent (wait 5 minutes)"
                    else:
                        reset_reason = "Unknown safety condition not met"
            
            # Get statistics if available
            stats = {}
            if hasattr(cb, 'blocked_count'):
                stats['blocked_orders'] = cb.blocked_count
            if hasattr(cb, 'allowed_count'):
                stats['allowed_orders'] = cb.allowed_count
            
            return {
                "status": "ok",
                "circuit_breaker": {
                    "active": is_active,
                    "state": state,
                    "failure_count": failure_count,
                    "threshold": threshold,
                    "last_failure": last_failure.isoformat() if last_failure else None,
                    "trigger_reason": trigger_reason,
                    "can_reset": can_reset,
                    "reset_blocked_reason": reset_reason if not can_reset else None,
                    "statistics": stats
                }
            }
        
        except Exception as e:
            logger.error(f"[CB_API] Error getting status: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get circuit breaker status: {str(e)}"
            )
    
    @app.post("/api/circuit-breaker/reset")
    async def reset_circuit_breaker(
        override: bool = False,
        reason: Optional[str] = None
    ):
        """
        Reset circuit breaker
        
        Args:
            override: Force reset even if conditions not safe (use with caution)
            reason: Explanation for manual reset (for audit trail)
        
        Returns:
            Status of reset operation
        
        Safety checks:
        - Requires failure count to be below threshold OR override=true
        - Requires last failure > 5 minutes ago OR override=true
        - Logs all reset attempts for audit
        """
        try:
            cb = circuit_breaker
            
            # Check if can reset
            can_reset = getattr(cb, 'can_reset', lambda: True)()
            
            if not can_reset and not override:
                failure_count = getattr(cb, 'failure_count', 0)
                threshold = getattr(cb, 'threshold', 0)
                last_failure = getattr(cb, 'last_failure_time', None)
                
                detail = {
                    "error": "Cannot reset circuit breaker - conditions not safe",
                    "failure_count": failure_count,
                    "threshold": threshold,
                    "last_failure": last_failure.isoformat() if last_failure else None,
                    "hint": "Use override=true to force reset (use with caution)"
                }
                
                raise HTTPException(status_code=400, detail=detail)
            
            # Perform reset
            reset_reason = reason or "Manual API reset"
            if override:
                reset_reason += " (OVERRIDE)"
            
            if hasattr(cb, 'reset'):
                cb.reset(reason=reset_reason)
            else:
                # Fallback: reset basic attributes
                cb.failure_count = 0
                cb.state = 'CLOSED'
                cb.last_failure_time = None
                cb.trigger_reason = None
            
            # Log reset
            logger.warning(
                f"[CB_API] Circuit breaker reset: reason='{reset_reason}', override={override}"
            )
            
            return {
                "status": "success",
                "action": "reset",
                "reason": reset_reason,
                "override_used": override,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[CB_API] Error resetting circuit breaker: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reset circuit breaker: {str(e)}"
            )
    
    @app.get("/api/circuit-breaker/history")
    async def get_circuit_breaker_history(limit: int = 50):
        """
        Get circuit breaker activation history
        
        Returns recent activation/reset events with timestamps and reasons
        """
        try:
            cb = circuit_breaker
            
            # Get history if available
            history = []
            if hasattr(cb, 'get_history'):
                history = cb.get_history(limit=limit)
            elif hasattr(cb, 'history'):
                history = cb.history[-limit:]
            
            return {
                "status": "ok",
                "history": history,
                "count": len(history)
            }
        
        except Exception as e:
            logger.error(f"[CB_API] Error getting history: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get circuit breaker history: {str(e)}"
            )
    
    logger.info("[CB_API] ✅ Circuit Breaker Management API registered")

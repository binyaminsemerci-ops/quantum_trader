"""
Learning Cadence API - REST interface for readiness checks.

Provides HTTP endpoints for querying learning readiness status.
Useful for manual checks and orchestration integration.
"""

import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from .cadence_policy import LearningCadencePolicy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Learning Cadence API",
    description="Intelligent gate-keeper for continuous learning",
    version="1.0.0"
)

# Global policy instance
policy = LearningCadencePolicy()


class TrainingCompletedRequest(BaseModel):
    """Request body for marking training complete"""
    action: str  # "calibration", "shadow", "promotion"
    notes: Optional[str] = None


@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "learning-cadence",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/readiness")
async def check_readiness():
    """
    Check learning readiness status.
    
    Returns complete evaluation with gates, triggers, and authorization.
    """
    try:
        result = policy.evaluate_learning_readiness()
        
        # Add metadata
        result["checked_at"] = datetime.now(timezone.utc).isoformat()
        result["mode"] = "logging_only"
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/readiness/simple")
async def check_readiness_simple():
    """
    Simplified readiness check - just yes/no.
    
    Returns:
        {"ready": true/false, "reason": "...", "actions": [...]}
    """
    try:
        result = policy.evaluate_learning_readiness()
        
        return {
            "ready": result["ready"],
            "reason": result["gate_reason"] if not result["gate_passed"] else result["trigger_reason"],
            "actions": result["allowed_actions"]
        }
    
    except Exception as e:
        logger.error(f"Simple readiness check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get current data statistics without full evaluation"""
    try:
        result = policy.evaluate_learning_readiness()
        return result["stats"]
    
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/completed")
async def mark_training_completed(request: TrainingCompletedRequest):
    """
    Mark that training was completed.
    
    This updates internal state (last training timestamp, trade count, etc.)
    """
    try:
        if request.action not in ["calibration", "shadow", "promotion"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. Must be one of: calibration, shadow, promotion"
            )
        
        policy.mark_training_completed(request.action)
        
        logger.info(f"✅ Training marked complete: action={request.action}, notes={request.notes}")
        
        return {
            "status": "recorded",
            "action": request.action,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark training complete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current policy configuration"""
    return {
        "gate": {
            "min_trades": policy.gate.min_trades,
            "min_days": policy.gate.min_days,
            "min_win_pct": policy.gate.min_win_pct,
            "min_loss_pct": policy.gate.min_loss_pct
        },
        "trigger": {
            "batch_size": policy.trigger.batch_size,
            "time_interval_hours": policy.trigger.time_interval_hours
        },
        "authorization": {
            "calibration_min_trades": policy.auth.calibration_min_trades,
            "shadow_min_trades": policy.auth.shadow_min_trades,
            "shadow_min_days": policy.auth.shadow_min_days,
            "promotion_min_trades": policy.auth.promotion_min_trades,
            "promotion_min_days": policy.auth.promotion_min_days
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)

"""
CLM API Routes - Monitoring and Control Endpoints

Provides REST API for CLM status, history, and manual triggers.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/clm", tags=["CLM"])


class CLMStatusResponse(BaseModel):
    """CLM status response model."""
    enabled: bool
    retrain_interval_days: int
    shadow_test_hours: int
    min_improvement_threshold: float
    active_models: dict
    last_retrain: Optional[str] = None
    next_retrain: Optional[str] = None


class ModelHistoryResponse(BaseModel):
    """Model version history response."""
    model_type: str
    versions: list[dict]


class TriggerRetrainRequest(BaseModel):
    """Manual retrain trigger request."""
    model_types: Optional[list[str]] = None
    force: bool = False


@router.get("/status", response_model=CLMStatusResponse)
async def get_clm_status():
    """
    Get CLM status and configuration.
    
    Returns current CLM configuration and active model versions.
    """
    try:
        from backend.main import app as app_instance
        
        clm = getattr(app_instance.state, 'clm', None)
        
        if not clm:
            return CLMStatusResponse(
                enabled=False,
                retrain_interval_days=0,
                shadow_test_hours=0,
                min_improvement_threshold=0.0,
                active_models={},
            )
        
        # Get active models
        active_models = {}
        for model_type in clm.model_types:
            active = clm.registry.get_active(model_type)
            if active:
                active_models[model_type.value] = {
                    "version": active.version,
                    "trained_at": active.trained_at.isoformat(),
                    "metrics": active.metrics,
                }
        
        return CLMStatusResponse(
            enabled=True,
            retrain_interval_days=clm.retrain_interval_days,
            shadow_test_hours=clm.shadow_test_hours,
            min_improvement_threshold=clm.min_improvement_threshold,
            active_models=active_models,
        )
        
    except Exception as e:
        logger.error(f"[CLM API] Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{model_type}", response_model=ModelHistoryResponse)
async def get_model_history(model_type: str, limit: int = 10):
    """
    Get model version history.
    
    Args:
        model_type: Model type (xgboost, lightgbm, nhits, patchtst)
        limit: Maximum number of versions to return
    
    Returns:
        List of model versions with metadata
    """
    try:
        from backend.main import app as app_instance
        from backend.services.ai.continuous_learning_manager import ModelType
        
        clm = getattr(app_instance.state, 'clm', None)
        if not clm:
            raise HTTPException(status_code=503, detail="CLM not available")
        
        # Validate model type
        try:
            mt = ModelType(model_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
        
        # Get history
        history = clm.registry.get_model_history(mt, limit=limit)
        
        versions = [
            {
                "version": artifact.version,
                "status": artifact.status.value,
                "trained_at": artifact.trained_at.isoformat(),
                "metrics": artifact.metrics,
                "data_points": artifact.data_points,
            }
            for artifact in history
        ]
        
        return ModelHistoryResponse(
            model_type=model_type,
            versions=versions,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CLM API] Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger")
async def trigger_retrain(request: TriggerRetrainRequest):
    """
    Manually trigger model retraining.
    
    Args:
        request: Trigger request with optional model types and force flag
    
    Returns:
        Retrain report
    """
    try:
        from backend.main import app as app_instance
        from backend.services.ai.continuous_learning_manager import ModelType
        
        clm = getattr(app_instance.state, 'clm', None)
        if not clm:
            raise HTTPException(status_code=503, detail="CLM not available")
        
        # Parse model types
        models = None
        if request.model_types:
            try:
                models = [ModelType(mt) for mt in request.model_types]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {e}")
        
        # Run retrain cycle
        logger.info(f"[CLM API] Manual retrain triggered (force={request.force})")
        
        report = clm.run_full_cycle(models=models, force=request.force)
        
        return {
            "success": True,
            "trigger": report.trigger.value,
            "triggered_at": report.triggered_at.isoformat(),
            "models_trained": [mt.value for mt in report.models_trained],
            "promoted_models": [mt.value for mt in report.promoted_models],
            "failed_models": [mt.value for mt in report.failed_models],
            "duration_seconds": report.total_duration_seconds,
            "summary": report.summary(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CLM API] Manual retrain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def clm_health():
    """
    CLM health check.
    
    Returns:
        Health status
    """
    try:
        from backend.main import app as app_instance
        
        clm = getattr(app_instance.state, 'clm', None)
        clm_task = getattr(app_instance.state, 'clm_task', None)
        
        if not clm:
            return {
                "status": "disabled",
                "message": "CLM not initialized"
            }
        
        task_running = clm_task is not None and not clm_task.done()
        
        return {
            "status": "healthy" if task_running else "degraded",
            "clm_enabled": True,
            "monitoring_loop_running": task_running,
            "model_types": [mt.value for mt in clm.model_types],
        }
        
    except Exception as e:
        logger.error(f"[CLM API] Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

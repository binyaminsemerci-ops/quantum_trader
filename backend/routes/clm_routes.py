"""FastAPI endpoints for Continuous Learning Manager.

Provides REST API for:
- Triggering manual retraining
- Checking retraining status
- Viewing model history
- Getting CLM configuration
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from backend.services.ai.continuous_learning_manager import (
    ContinuousLearningManager,
    ModelType,
    RetrainTrigger,
    RetrainReport,
)
from backend.services.clm_implementations import (
    BinanceDataClient,
    QuantumFeatureEngineer,
    QuantumModelTrainer,
    QuantumModelEvaluator,
    QuantumShadowTester,
    SQLModelRegistry,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clm", tags=["CLM"])


# ============================================================================
# GLOBAL CLM INSTANCE (LAZY INITIALIZATION)
# ============================================================================

_clm_instance: Optional[ContinuousLearningManager] = None
_last_report: Optional[RetrainReport] = None


def get_clm() -> ContinuousLearningManager:
    """Get or create global CLM instance."""
    global _clm_instance
    
    if _clm_instance is None:
        logger.info("Initializing CLM instance...")
        
        # Initialize all components
        data_client = BinanceDataClient(symbol="BTCUSDT", interval="1h")
        feature_engineer = QuantumFeatureEngineer(use_advanced=True)
        trainer = QuantumModelTrainer()
        evaluator = QuantumModelEvaluator(feature_engineer=feature_engineer)
        shadow_tester = QuantumShadowTester(
            data_client=data_client,
            feature_engineer=feature_engineer
        )
        registry = SQLModelRegistry()
        
        _clm_instance = ContinuousLearningManager(
            data_client=data_client,
            feature_engineer=feature_engineer,
            trainer=trainer,
            evaluator=evaluator,
            shadow_tester=shadow_tester,
            registry=registry,
            retrain_interval_days=7,
            shadow_test_hours=24,
            min_improvement_threshold=0.02,
            training_lookback_days=90,
        )
        
        logger.info("CLM initialized successfully")
    
    return _clm_instance


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RetrainRequest(BaseModel):
    """Request to trigger retraining."""
    models: Optional[List[str]] = Field(
        None,
        description="List of model types to retrain (xgboost, lightgbm, nhits, patchtst). If None, retrains all."
    )
    force: bool = Field(
        False,
        description="Force retraining even if no triggers detected"
    )


class RetrainStatusResponse(BaseModel):
    """Response with retraining status."""
    status: str
    message: str
    report: Optional[Dict[str, Any]] = None


class ModelStatusResponse(BaseModel):
    """Response with current model status."""
    models: Dict[str, Dict[str, Any]]


class TriggerStatusResponse(BaseModel):
    """Response with trigger check results."""
    triggers: Dict[str, Optional[str]]
    triggered_count: int


class ModelHistoryResponse(BaseModel):
    """Response with model version history."""
    model_type: str
    versions: List[Dict[str, Any]]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/retrain", response_model=RetrainStatusResponse)
async def trigger_retraining(
    request: RetrainRequest,
    background_tasks: BackgroundTasks
) -> RetrainStatusResponse:
    """Trigger manual retraining of models.
    
    This endpoint starts a background retraining job. Use GET /clm/status
    to check progress.
    
    Example:
        POST /api/clm/retrain
        {
            "models": ["xgboost", "lightgbm"],
            "force": true
        }
    """
    try:
        clm = get_clm()
        
        # Parse model types
        model_types = None
        if request.models:
            model_types = [ModelType(m.upper()) for m in request.models]
        
        # Run retraining in background
        def run_retraining():
            global _last_report
            try:
                logger.info(f"Starting retraining: models={request.models}, force={request.force}")
                _last_report = clm.run_full_cycle(models=model_types, force=request.force)
                logger.info(f"Retraining complete: promoted={len(_last_report.promoted_models)}")
            except Exception as e:
                logger.error(f"Retraining failed: {e}", exc_info=True)
        
        background_tasks.add_task(run_retraining)
        
        return RetrainStatusResponse(
            status="started",
            message="Retraining job started in background",
            report=None
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=RetrainStatusResponse)
async def get_retraining_status() -> RetrainStatusResponse:
    """Get status of last retraining job.
    
    Returns the most recent RetrainReport if available.
    
    Example:
        GET /api/clm/status
    """
    try:
        global _last_report
        
        if _last_report is None:
            return RetrainStatusResponse(
                status="idle",
                message="No retraining jobs have been run yet",
                report=None
            )
        
        # Convert report to dict
        report_dict = {
            "trigger": _last_report.trigger.value if _last_report.trigger else None,
            "started_at": _last_report.started_at.isoformat(),
            "models_trained": [m.value for m in _last_report.models_trained],
            "promoted_models": [m.value for m in _last_report.promoted_models],
            "failed_models": [m.value for m in _last_report.failed_models],
            "total_duration_seconds": _last_report.total_duration_seconds,
            "evaluations": {
                model_type.value: {
                    "rmse": eval_result.rmse,
                    "mae": eval_result.mae,
                    "directional_accuracy": eval_result.directional_accuracy,
                    "vs_active_rmse_delta": eval_result.vs_active_rmse_delta,
                    "vs_active_direction_delta": eval_result.vs_active_direction_delta,
                }
                for model_type, eval_result in _last_report.evaluations.items()
            },
            "shadow_tests": {
                model_type.value: {
                    "live_predictions": shadow_result.live_predictions,
                    "candidate_mae": shadow_result.candidate_mae,
                    "active_mae": shadow_result.active_mae,
                    "recommend_promotion": shadow_result.recommend_promotion,
                    "reason": shadow_result.reason,
                }
                for model_type, shadow_result in _last_report.shadow_tests.items()
            },
        }
        
        return RetrainStatusResponse(
            status="completed",
            message=f"Last run: {_last_report.started_at.isoformat()}",
            report=report_dict
        )
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """Get current status of all active models.
    
    Returns version, training date, and metrics for each model type.
    
    Example:
        GET /api/clm/models
    """
    try:
        clm = get_clm()
        status = clm.get_model_status_summary()
        
        return ModelStatusResponse(models=status)
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/triggers", response_model=TriggerStatusResponse)
async def check_triggers() -> TriggerStatusResponse:
    """Check which models need retraining.
    
    Returns trigger status for each model type.
    
    Example:
        GET /api/clm/triggers
    """
    try:
        clm = get_clm()
        triggers = clm.check_if_retrain_needed()
        
        # Convert to dict
        trigger_dict = {
            model_type.value: trigger.value if trigger else None
            for model_type, trigger in triggers.items()
        }
        
        triggered_count = sum(1 for t in triggers.values() if t)
        
        return TriggerStatusResponse(
            triggers=trigger_dict,
            triggered_count=triggered_count
        )
        
    except Exception as e:
        logger.error(f"Failed to check triggers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{model_type}", response_model=ModelHistoryResponse)
async def get_model_history(
    model_type: str,
    limit: int = 10
) -> ModelHistoryResponse:
    """Get version history for a specific model type.
    
    Args:
        model_type: Model type (xgboost, lightgbm, nhits, patchtst)
        limit: Maximum number of versions to return (default: 10)
    
    Example:
        GET /api/clm/history/xgboost?limit=5
    """
    try:
        clm = get_clm()
        
        # Parse model type
        mt = ModelType(model_type.upper())
        
        # Get history from registry
        artifacts = clm.registry.get_model_history(mt, limit=limit)
        
        # Convert to dict
        versions = [
            {
                "version": artifact.version,
                "status": artifact.status.value,
                "trained_at": artifact.trained_at.isoformat(),
                "metrics": artifact.metrics,
                "data_points": artifact.data_points,
            }
            for artifact in artifacts
        ]
        
        return ModelHistoryResponse(
            model_type=model_type,
            versions=versions
        )
        
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type: {model_type}. Must be one of: xgboost, lightgbm, nhits, patchtst"
        )
    except Exception as e:
        logger.error(f"Failed to get model history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current CLM configuration.
    
    Returns all configuration parameters.
    
    Example:
        GET /api/clm/config
    """
    try:
        clm = get_clm()
        
        return {
            "retrain_interval_days": clm.retrain_interval_days,
            "shadow_test_hours": clm.shadow_test_hours,
            "min_improvement_threshold": clm.min_improvement_threshold,
            "training_lookback_days": clm.training_lookback_days,
            "supported_models": [m.value for m in ModelType],
        }
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-cache")
async def clear_cache() -> Dict[str, str]:
    """Clear CLM cache and reset instance.
    
    Forces reinitialization on next request.
    
    Example:
        DELETE /api/clm/clear-cache
    """
    try:
        global _clm_instance, _last_report
        
        _clm_instance = None
        _last_report = None
        
        logger.info("CLM cache cleared")
        
        return {
            "status": "ok",
            "message": "CLM cache cleared. Will reinitialize on next request."
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

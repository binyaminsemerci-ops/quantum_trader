"""
REST API for RL Training Service

Endpoints for monitoring training status, triggering manual retrains,
viewing shadow models, and drift metrics.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional

from microservices.rl_training.models import (
    ModelType,
    TrainingTrigger,
    TrainingJobInfo,
    ModelVersionInfo,
    ShadowModelStatus,
    DriftMetrics,
    ServiceHealth,
    TrainingConfig,
)


router = APIRouter(prefix="/api/training", tags=["training"])


# Dependency to get service components
# (Will be injected via app.state in main.py)
def get_training_daemon():
    from microservices.rl_training.main import app
    return app.state.training_daemon


def get_clm():
    from microservices.rl_training.main import app
    return app.state.clm


def get_shadow_manager():
    from microservices.rl_training.main import app
    return app.state.shadow_manager


def get_drift_detector():
    from microservices.rl_training.main import app
    return app.state.drift_detector


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@router.get("/health", response_model=ServiceHealth)
async def health_check(
    training_daemon=Depends(get_training_daemon)
) -> ServiceHealth:
    """Get service health status"""
    from datetime import datetime, timezone
    from microservices.rl_training.main import service_start_time
    
    uptime_seconds = (datetime.now(timezone.utc) - service_start_time).total_seconds()
    
    # Get last training job
    history = training_daemon.get_training_history(limit=1)
    last_job = None
    if history:
        job = history[0]
        last_job = TrainingJobInfo(
            job_id=job["job_id"],
            model_type=ModelType(job["model_type"]),
            status=job["status"],
            trigger=TrainingTrigger(job["trigger"]),
            started_at=job["started_at"],
            completed_at=job.get("completed_at"),
            duration_seconds=job.get("duration_seconds")
        )
    
    return ServiceHealth(
        service="rl-training",
        version="1.0.0",
        status="healthy",
        uptime_seconds=uptime_seconds,
        components=[
            {"name": "TrainingDaemon", "status": "running"},
            {"name": "CLM", "status": "running"},
            {"name": "ShadowModelManager", "status": "running"},
            {"name": "DriftDetector", "status": "running"},
            {"name": "EventBus", "status": "connected"},
        ],
        last_training_job=last_job
    )


# ============================================================================
# TRAINING JOBS
# ============================================================================

@router.get("/jobs/history")
async def get_training_history(
    limit: int = 50,
    training_daemon=Depends(get_training_daemon)
) -> Dict[str, Any]:
    """Get training job history"""
    history = training_daemon.get_training_history(limit=limit)
    
    return {
        "total": len(history),
        "jobs": history
    }


@router.get("/jobs/current")
async def get_current_job(
    training_daemon=Depends(get_training_daemon)
) -> Dict[str, Any]:
    """Get currently running job"""
    job_id = training_daemon.get_current_job_id()
    
    if job_id is None:
        return {"status": "idle", "job_id": None}
    
    return {"status": "running", "job_id": job_id}


@router.post("/jobs/trigger")
async def trigger_training(
    model_type: ModelType,
    reason: str = "Manual trigger",
    training_daemon=Depends(get_training_daemon)
) -> Dict[str, Any]:
    """Manually trigger training job"""
    result = await training_daemon.run_training_cycle(
        model_type=model_type,
        trigger=TrainingTrigger.MANUAL,
        reason=reason
    )
    
    return result


# ============================================================================
# CLM (Continuous Learning Manager)
# ============================================================================

@router.get("/clm/status")
async def get_clm_status(
    clm=Depends(get_clm)
) -> Dict[str, Any]:
    """Get CLM status"""
    last_retrain_times = clm.get_last_retrain_times()
    needs_retrain = await clm.check_if_retrain_needed()
    
    return {
        "last_retrain_times": last_retrain_times,
        "needs_retrain": {
            model_type.value: should_retrain
            for model_type, should_retrain in needs_retrain.items()
        }
    }


@router.post("/clm/run-cycle")
async def run_clm_cycle(
    clm=Depends(get_clm)
) -> Dict[str, Any]:
    """Run full CLM cycle"""
    result = await clm.run_full_cycle()
    return result


# ============================================================================
# SHADOW MODELS
# ============================================================================

@router.get("/shadow/models", response_model=List[ShadowModelStatus])
async def get_shadow_models(
    shadow_manager=Depends(get_shadow_manager)
) -> List[ShadowModelStatus]:
    """Get all shadow models"""
    return shadow_manager.get_shadow_models()


@router.get("/shadow/champion")
async def get_champion_model(
    shadow_manager=Depends(get_shadow_manager)
) -> Dict[str, Any]:
    """Get current champion model"""
    champion = shadow_manager.get_champion_model()
    
    if champion is None:
        return {"champion": None}
    
    return {"champion": champion}


@router.post("/shadow/register")
async def register_shadow_model(
    model_type: ModelType,
    version: str,
    model_name: Optional[str] = None,
    shadow_manager=Depends(get_shadow_manager)
) -> Dict[str, Any]:
    """Register new shadow model"""
    model_name = await shadow_manager.register_shadow_model(
        model_type=model_type,
        version=version,
        model_name=model_name
    )
    
    return {"status": "registered", "model_name": model_name}


@router.post("/shadow/evaluate")
async def evaluate_shadow_model(
    model_name: str,
    shadow_metrics: Dict[str, float],
    champion_metrics: Dict[str, float],
    shadow_manager=Depends(get_shadow_manager)
) -> Dict[str, Any]:
    """Evaluate shadow model vs champion"""
    result = await shadow_manager.evaluate_shadow_model(
        model_name=model_name,
        shadow_metrics=shadow_metrics,
        champion_metrics=champion_metrics
    )
    
    return result


@router.post("/shadow/promote")
async def promote_shadow_model(
    model_name: str,
    reason: str = "Manual promotion",
    shadow_manager=Depends(get_shadow_manager)
) -> Dict[str, Any]:
    """Promote shadow model to active champion"""
    result = await shadow_manager.promote_shadow_to_active(
        model_name=model_name,
        reason=reason
    )
    
    return result


# ============================================================================
# DRIFT DETECTION
# ============================================================================

@router.get("/drift/history")
async def get_drift_history(
    limit: int = 100,
    drift_detector=Depends(get_drift_detector)
) -> Dict[str, Any]:
    """Get drift detection history"""
    history = drift_detector.get_drift_history(limit=limit)
    
    return {
        "total": len(history),
        "drift_events": history
    }


@router.get("/drift/distributions")
async def get_current_distributions(
    drift_detector=Depends(get_drift_detector)
) -> Dict[str, Any]:
    """Get current feature distributions"""
    distributions = drift_detector.get_current_distributions()
    
    return {
        "distributions": distributions
    }


@router.post("/drift/check-feature")
async def check_feature_drift(
    feature_name: str,
    current_distribution: Dict[str, Any],
    drift_detector=Depends(get_drift_detector)
) -> Dict[str, Any]:
    """Check drift for a specific feature"""
    result = await drift_detector.check_feature_drift(
        feature_name=feature_name,
        current_distribution=current_distribution
    )
    
    return result

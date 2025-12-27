"""
API Endpoints for ML/AI Pipeline Management.

Provides REST endpoints for:
- Model management (list, get, promote, retire)
- Shadow testing status
- Retraining triggers
- Drift monitoring
- CLM system status
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from backend.core.database import get_db_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    def get_db_session():
        raise HTTPException(status_code=503, detail="Database not available")

from backend.core.event_bus import get_event_bus
from backend.core.policy_store import get_policy_store
from backend.domains.learning.clm import ContinuousLearningManager, CLMConfig
from backend.domains.learning.model_registry import ModelType, ModelStatus
from backend.domains.learning.retraining import RetrainingType

# Create router
router = APIRouter(prefix="/api/v1/learning", tags=["ML/AI Learning"])

# Global CLM instance (initialized on startup)
_clm_instance: Optional[ContinuousLearningManager] = None


# ============================================================================
# Request/Response Models
# ============================================================================

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    version: str
    status: str
    metrics: dict
    created_at: str
    promoted_at: Optional[str] = None
    feature_count: Optional[int] = None


class RetrainingRequest(BaseModel):
    retraining_type: str = "full"  # full, partial, incremental
    model_types: Optional[List[str]] = None
    trigger_reason: str = "manual"
    days_of_data: int = 90


class RetrainingJobStatus(BaseModel):
    job_id: str
    status: str
    retraining_type: str
    trigger_reason: str
    models_trained: int
    models_succeeded: int
    models_failed: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ShadowTestSummary(BaseModel):
    model_id: str
    model_type: str
    total_predictions: int
    predictions_with_outcomes: int
    avg_error: Optional[float] = None
    first_prediction: Optional[str] = None
    last_prediction: Optional[str] = None


class DriftEvent(BaseModel):
    id: int
    drift_type: str
    severity: str
    drift_score: float
    p_value: float
    model_type: Optional[str] = None
    feature_name: Optional[str] = None
    detection_time: str
    trigger_retraining: bool
    notes: Optional[str] = None


class CLMStatus(BaseModel):
    running: bool
    last_retraining: Optional[str] = None
    last_drift_check: Optional[str] = None
    last_performance_check: Optional[str] = None
    config: dict
    active_models: dict
    shadow_models: dict


class PromoteRequest(BaseModel):
    model_type: str


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_clm() -> ContinuousLearningManager:
    """Get CLM instance."""
    if _clm_instance is None:
        raise HTTPException(status_code=503, detail="CLM not initialized")
    return _clm_instance


async def initialize_clm(
    db: AsyncSession,
    event_bus,
    policy_store,
    config: Optional[CLMConfig] = None,
):
    """Initialize CLM on app startup."""
    global _clm_instance
    
    if _clm_instance is None:
        from backend.domains.learning.clm import create_clm
        _clm_instance = await create_clm(db, event_bus, policy_store, config)
        await _clm_instance.start()


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """
    List models with optional filters.
    
    Query params:
    - model_type: Filter by model type (xgboost, lightgbm, nhits, patchtst)
    - status: Filter by status (training, shadow, active, retired)
    - limit: Max results (default 50)
    """
    model_type_enum = ModelType(model_type) if model_type else None
    status_enum = ModelStatus(status) if status else None
    
    models = await clm.model_registry.list_models(
        model_type=model_type_enum,
        status=status_enum,
        limit=limit,
    )
    
    return [
        ModelInfo(
            model_id=m.model_id,
            model_type=m.model_type.value,
            version=m.version,
            status=m.status.value,
            metrics=m.metrics,
            created_at=m.created_at.isoformat(),
            promoted_at=m.promoted_at.isoformat() if m.promoted_at else None,
            feature_count=m.feature_count,
        )
        for m in models
    ]


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Get details for a specific model."""
    model = await clm.model_registry.get_model(model_id, load_object=False)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return ModelInfo(
        model_id=model.model_id,
        model_type=model.model_type.value,
        version=model.version,
        status=model.status.value,
        metrics=model.metrics,
        created_at=model.created_at.isoformat(),
        promoted_at=model.promoted_at.isoformat() if model.promoted_at else None,
        feature_count=model.feature_count,
    )


@router.post("/models/{model_id}/retire")
async def retire_model(
    model_id: str,
    reason: Optional[str] = None,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Retire a model."""
    success = await clm.model_registry.retire_model(model_id, reason)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {"status": "success", "message": f"Model {model_id} retired"}


@router.post("/models/promote", response_model=dict)
async def promote_shadow_model(
    request: PromoteRequest,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Manually promote shadow model to active."""
    try:
        model_type = ModelType(request.model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {request.model_type}")
    
    success = await clm.manual_promote_shadow(model_type)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"No shadow model found for {request.model_type}")
    
    return {
        "status": "success",
        "message": f"Shadow model promoted for {request.model_type}",
    }


# ============================================================================
# Retraining Endpoints
# ============================================================================

@router.post("/retraining/trigger", response_model=dict)
async def trigger_retraining(
    request: RetrainingRequest,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """
    Trigger model retraining.
    
    Body:
    - retraining_type: "full", "partial", or "incremental"
    - model_types: Optional list of model types to retrain
    - trigger_reason: Reason for retraining (default: "manual")
    - days_of_data: Number of days of training data (default: 90)
    """
    try:
        retraining_type = RetrainingType(request.retraining_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid retraining_type: {request.retraining_type}")
    
    model_types = None
    if request.model_types:
        try:
            model_types = [ModelType(mt) for mt in request.model_types]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {e}")
    
    job_id = await clm.trigger_retraining(
        retraining_type=retraining_type,
        model_types=model_types,
        trigger_reason=request.trigger_reason,
    )
    
    return {
        "status": "success",
        "job_id": job_id,
        "message": "Retraining job started",
    }


@router.get("/retraining/{job_id}", response_model=RetrainingJobStatus)
async def get_retraining_status(
    job_id: str,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Get status of a retraining job."""
    job_status = await clm.retraining_orchestrator.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return RetrainingJobStatus(**job_status)


# ============================================================================
# Shadow Testing Endpoints
# ============================================================================

@router.get("/shadow-testing/summary", response_model=List[ShadowTestSummary])
async def get_shadow_testing_summary(
    model_type: Optional[str] = None,
    days: int = 30,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Get shadow testing summary."""
    model_type_enum = ModelType(model_type) if model_type else None
    
    summaries = await clm.shadow_tester.get_shadow_test_summary(
        model_type=model_type_enum,
        days=days,
    )
    
    return [ShadowTestSummary(**s) for s in summaries]


# ============================================================================
# Drift Monitoring Endpoints
# ============================================================================

@router.get("/drift/events", response_model=List[DriftEvent])
async def get_drift_events(
    days: int = 7,
    drift_type: Optional[str] = None,
    model_type: Optional[str] = None,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Get recent drift events."""
    from backend.domains.learning.drift_detector import DriftType as DriftTypeEnum
    
    drift_type_enum = DriftTypeEnum(drift_type) if drift_type else None
    model_type_enum = ModelType(model_type) if model_type else None
    
    events = await clm.drift_detector.get_recent_drift_events(
        days=days,
        drift_type=drift_type_enum,
        model_type=model_type_enum,
    )
    
    return [DriftEvent(**e) for e in events]


@router.post("/drift/check/{model_type}", response_model=dict)
async def trigger_drift_check(
    model_type: str,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Manually trigger drift check for a model."""
    try:
        model_type_enum = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")
    
    result = await clm.manual_trigger_drift_check(model_type_enum)
    
    return result


# ============================================================================
# CLM System Status Endpoints
# ============================================================================

@router.get("/status", response_model=CLMStatus)
async def get_clm_status(
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Get overall CLM system status."""
    status = await clm.get_system_status()
    return CLMStatus(**status)


@router.post("/start")
async def start_clm(
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Start CLM (if stopped)."""
    if clm.running:
        return {"status": "already_running"}
    
    await clm.start()
    return {"status": "started"}


@router.post("/stop")
async def stop_clm(
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Stop CLM."""
    if not clm.running:
        return {"status": "already_stopped"}
    
    await clm.stop()
    return {"status": "stopped"}


# ============================================================================
# Performance Monitoring Endpoints
# ============================================================================

@router.get("/performance/{model_id}")
async def get_model_performance(
    model_id: str,
    days: int = 30,
    clm: ContinuousLearningManager = Depends(get_clm),
):
    """Get performance history for a model."""
    history = await clm.model_supervisor.get_performance_history(model_id, days)
    
    if not history:
        raise HTTPException(status_code=404, detail=f"No performance data for {model_id}")
    
    return {"model_id": model_id, "history": history}


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml_pipeline",
        "timestamp": datetime.utcnow().isoformat(),
    }

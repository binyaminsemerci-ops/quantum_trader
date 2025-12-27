"""
CLM v3 FastAPI App - REST API for Continuous Learning Manager v3.

Endpoints:
- GET /health - Health check
- GET /clm/status - CLM status and statistics
- POST /clm/train - Manually trigger training job
- POST /clm/promote - Manually promote model
- POST /clm/rollback - Rollback to previous model version
- GET /clm/jobs - List training jobs
- GET /clm/models - List model versions
- GET /clm/candidates - List strategy candidates
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from backend.services.clm_v3.models import (
    ModelQuery,
    ModelStatus,
    ModelType,
    PromotionRequest,
    RollbackRequest,
    TrainingJob,
    TriggerReason,
)
from backend.services.clm_v3.orchestrator import ClmOrchestrator
from backend.services.clm_v3.scheduler import TrainingScheduler
from backend.services.clm_v3.storage import ModelRegistryV3
from backend.services.clm_v3.strategies import StrategyEvolutionEngine

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CLM v3 - Continuous Learning Manager v3",
    description="Model training orchestration, promotion, and strategy evolution",
    version="3.0.0",
)

# Global instances (set by main.py)
registry: Optional[ModelRegistryV3] = None
orchestrator: Optional[ClmOrchestrator] = None
scheduler: Optional[TrainingScheduler] = None
evolution: Optional[StrategyEvolutionEngine] = None


def set_dependencies(
    registry_instance: ModelRegistryV3,
    orchestrator_instance: ClmOrchestrator,
    scheduler_instance: TrainingScheduler,
    evolution_instance: StrategyEvolutionEngine,
):
    """Set global dependency instances."""
    global registry, orchestrator, scheduler, evolution
    registry = registry_instance
    orchestrator = orchestrator_instance
    scheduler = scheduler_instance
    evolution = evolution_instance


# ============================================================================
# Request/Response Models
# ============================================================================

class ManualTrainingRequest(BaseModel):
    """Request to manually trigger training."""
    
    model_type: ModelType
    symbol: Optional[str] = None
    timeframe: str = "1h"
    dataset_span_days: int = 90
    training_params: Optional[Dict] = None


class TrainingJobResponse(BaseModel):
    """Training job response."""
    
    job_id: UUID
    model_type: ModelType
    status: str
    trigger_reason: TriggerReason
    triggered_by: str
    created_at: str


# ============================================================================
# Health & Status
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "clm_v3",
        "version": "3.0.0",
    }


@app.get("/clm/status")
async def get_clm_status():
    """
    Get CLM v3 status and statistics.
    
    Returns:
        - Registry stats (models, jobs, evaluations)
        - Scheduler status (next training times)
        - Evolution stats (candidates)
    """
    if not all([registry, scheduler, evolution]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 not initialized",
        )
    
    return {
        "service": "clm_v3",
        "status": "running",
        "registry": registry.get_stats(),
        "scheduler": scheduler.get_status(),
        "evolution": evolution.get_stats(),
    }


# ============================================================================
# Training Jobs
# ============================================================================

@app.post("/clm/train", response_model=TrainingJobResponse)
async def trigger_training(request: ManualTrainingRequest):
    """
    Manually trigger a training job.
    
    Args:
        request: ManualTrainingRequest with training parameters
    
    Returns:
        TrainingJobResponse with job details
    """
    if not scheduler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 scheduler not initialized",
        )
    
    logger.info(
        f"[CLM v3 API] Manual training request: {request.model_type.value} "
        f"(symbol={request.symbol}, timeframe={request.timeframe})"
    )
    
    # Create training job via scheduler
    job = await scheduler.trigger_training(
        model_type=request.model_type,
        trigger_reason=TriggerReason.MANUAL,
        triggered_by="api_manual",
        symbol=request.symbol,
        timeframe=request.timeframe,
        dataset_span_days=request.dataset_span_days,
        training_params=request.training_params,
    )
    
    # Start training in background (non-blocking)
    import asyncio
    asyncio.create_task(orchestrator.handle_training_job(job))
    
    return TrainingJobResponse(
        job_id=job.id,
        model_type=job.model_type,
        status=job.status,
        trigger_reason=job.trigger_reason,
        triggered_by=job.triggered_by,
        created_at=job.triggered_at.isoformat(),
    )


@app.get("/clm/jobs")
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    List training jobs.
    
    Args:
        status: Filter by status (pending, in_progress, completed, failed)
        limit: Max results
    
    Returns:
        List of training jobs
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 registry not initialized",
        )
    
    jobs = registry.list_training_jobs(status=status, limit=limit)
    
    return {
        "jobs": [
            {
                "job_id": str(job.id),
                "model_type": job.model_type.value,
                "status": job.status,
                "trigger_reason": job.trigger_reason.value,
                "triggered_by": job.triggered_by,
                "triggered_at": job.triggered_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message,
            }
            for job in jobs
        ],
        "total": len(jobs),
    }


@app.get("/clm/jobs/{job_id}")
async def get_training_job(job_id: UUID):
    """Get training job by ID."""
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 registry not initialized",
        )
    
    job = registry.get_training_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found",
        )
    
    return job.dict()


# ============================================================================
# Model Management
# ============================================================================

@app.get("/clm/models")
async def list_models(
    model_type: Optional[ModelType] = None,
    status: Optional[ModelStatus] = None,
    min_sharpe: Optional[float] = None,
    limit: int = 50,
):
    """
    List model versions with filtering.
    
    Args:
        model_type: Filter by model type
        status: Filter by status
        min_sharpe: Minimum Sharpe ratio
        limit: Max results
    
    Returns:
        List of model versions
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 registry not initialized",
        )
    
    query = ModelQuery(
        model_type=model_type,
        status=status,
        min_sharpe=min_sharpe,
        limit=limit,
    )
    
    models = registry.query_models(query)
    
    return {
        "models": [
            {
                "model_id": m.model_id,
                "version": m.version,
                "model_type": m.model_type.value,
                "status": m.status.value,
                "created_at": m.created_at.isoformat(),
                "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
                "train_metrics": m.train_metrics,
                "validation_metrics": m.validation_metrics,
            }
            for m in models
        ],
        "total": len(models),
    }


@app.post("/clm/promote")
async def promote_model(request: PromotionRequest):
    """
    Manually promote a model to production.
    
    Args:
        request: PromotionRequest with model_id, version, reason
    
    Returns:
        Success response
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 registry not initialized",
        )
    
    logger.info(
        f"[CLM v3 API] Manual promotion request: {request.model_id} v{request.version} "
        f"(by {request.promoted_by}, reason: {request.reason})"
    )
    
    # Promote model
    success = registry.promote_model(
        model_id=request.model_id,
        version=request.version,
        promoted_by=request.promoted_by,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to promote {request.model_id} v{request.version}",
        )
    
    return {
        "success": True,
        "model_id": request.model_id,
        "version": request.version,
        "status": "promoted",
        "promoted_by": request.promoted_by,
    }


@app.post("/clm/rollback")
async def rollback_model(request: RollbackRequest):
    """
    Rollback to a previous model version.
    
    Args:
        request: RollbackRequest with model_id, target_version, reason
    
    Returns:
        Success response
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 registry not initialized",
        )
    
    logger.warning(
        f"[CLM v3 API] ROLLBACK request: {request.model_id} → v{request.target_version} "
        f"(by {request.rollback_by}, reason: {request.reason})"
    )
    
    # Rollback model
    success = registry.rollback_to_version(
        model_id=request.model_id,
        target_version=request.target_version,
        reason=request.reason,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to rollback {request.model_id} to v{request.target_version}",
        )
    
    return {
        "success": True,
        "model_id": request.model_id,
        "version": request.target_version,
        "status": "rolled_back",
        "rollback_by": request.rollback_by,
        "reason": request.reason,
    }


# ============================================================================
# Strategy Evolution
# ============================================================================

@app.get("/clm/candidates")
async def list_strategy_candidates(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    List strategy candidates.
    
    Args:
        status: Filter by status
        limit: Max results
    
    Returns:
        List of strategy candidates
    """
    if not evolution:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLM v3 evolution engine not initialized",
        )
    
    from backend.services.clm_v3.strategies import StrategyStatus
    
    status_enum = StrategyStatus(status) if status else None
    candidates = evolution.list_candidates(status=status_enum)
    
    return {
        "candidates": [
            {
                "id": str(c.id),
                "base_strategy": c.base_strategy,
                "model_type": c.model_type.value,
                "params": c.params,
                "origin": c.origin.value,
                "status": c.status.value,
                "created_at": c.created_at.isoformat(),
                "fitness_score": c.fitness_score,
                "performance_metrics": c.performance_metrics,
            }
            for c in candidates[:limit]
        ],
        "total": len(candidates),
    }


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    """Startup event - log initialization."""
    logger.info("[CLM v3 API] FastAPI app started")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown event - cleanup."""
    logger.info("[CLM v3 API] FastAPI app shutting down")
    
    if scheduler and scheduler._running:
        await scheduler.stop()

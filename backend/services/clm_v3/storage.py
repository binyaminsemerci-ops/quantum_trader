"""
CLM v3 Storage - Model Registry with versioning and metadata.

Provides:
- Model registration and versioning
- Promotion/rollback workflows
- Query and filtering
- Persistence (file-based + metadata in memory/DB)
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from backend.services.clm_v3.models import (
    EvaluationResult,
    ModelQuery,
    ModelStatus,
    ModelType,
    ModelVersion,
    TrainingJob,
)

logger = logging.getLogger(__name__)


class ModelRegistryV3:
    """
    Model Registry v3 - Centralized model storage and versioning.
    
    Features:
    - Model version tracking
    - Promotion/rollback workflows
    - Metadata persistence
    - Query/filtering
    """
    
    def __init__(
        self,
        models_dir: str = "/app/models",
        metadata_dir: str = "/app/data/clm_v3/registry",
    ):
        """
        Initialize Model Registry.
        
        Args:
            models_dir: Directory for storing model artifacts
            metadata_dir: Directory for storing metadata JSON files
        """
        self.models_dir = Path(models_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # {model_id: {version: ModelVersion}}
        self.training_jobs: Dict[UUID, TrainingJob] = {}
        self.evaluations: Dict[str, List[EvaluationResult]] = {}  # {model_id_version: [results]}
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"[CLM v3 Registry] Initialized (models={len(self.models)}, jobs={len(self.training_jobs)})")
    
    # ========================================================================
    # Training Jobs
    # ========================================================================
    
    def register_training_job(self, job: TrainingJob) -> TrainingJob:
        """
        Register a new training job.
        
        Args:
            job: TrainingJob to register
        
        Returns:
            Registered TrainingJob with ID
        """
        self.training_jobs[job.id] = job
        self._save_training_job(job)
        
        logger.info(f"[CLM v3 Registry] Registered training job {job.id} ({job.model_type}, trigger={job.trigger_reason})")
        
        return job
    
    def update_training_job(self, job_id: UUID, updates: Dict) -> Optional[TrainingJob]:
        """Update training job status."""
        if job_id not in self.training_jobs:
            logger.warning(f"[CLM v3 Registry] Training job {job_id} not found")
            return None
        
        job = self.training_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        
        self._save_training_job(job)
        return job
    
    def get_training_job(self, job_id: UUID) -> Optional[TrainingJob]:
        """Get training job by ID."""
        return self.training_jobs.get(job_id)
    
    def list_training_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        jobs = list(self.training_jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by triggered_at (newest first)
        jobs.sort(key=lambda j: j.triggered_at, reverse=True)
        
        return jobs[:limit]
    
    # ========================================================================
    # Model Versions
    # ========================================================================
    
    def register_model_version(self, model: ModelVersion) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model: ModelVersion to register
        
        Returns:
            Registered ModelVersion
        """
        if model.model_id not in self.models:
            self.models[model.model_id] = {}
        
        self.models[model.model_id][model.version] = model
        self._save_model_metadata(model)
        
        logger.info(
            f"[CLM v3 Registry] Registered model {model.model_id} v{model.version} "
            f"(status={model.status}, size={model.model_size_bytes} bytes)"
        )
        
        return model
    
    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        return self.models.get(model_id, {}).get(version)
    
    def list_model_versions(
        self,
        model_id: str,
        status: Optional[ModelStatus] = None,
    ) -> List[ModelVersion]:
        """List all versions of a model."""
        if model_id not in self.models:
            return []
        
        versions = list(self.models[model_id].values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        # Sort by created_at (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return versions
    
    def get_production_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get current production model version."""
        versions = self.list_model_versions(model_id, status=ModelStatus.PRODUCTION)
        return versions[0] if versions else None
    
    def query_models(self, query: ModelQuery) -> List[ModelVersion]:
        """Query models with filtering."""
        results = []
        
        for model_id, versions in self.models.items():
            for version in versions.values():
                # Apply filters
                if query.model_type and version.model_type != query.model_type:
                    continue
                if query.status and version.status != query.status:
                    continue
                if query.created_after and version.created_at < query.created_after:
                    continue
                if query.created_before and version.created_at > query.created_before:
                    continue
                if query.min_sharpe:
                    sharpe = version.validation_metrics.get("sharpe_ratio", 0)
                    if sharpe < query.min_sharpe:
                        continue
                
                results.append(version)
        
        # Sort by created_at (newest first)
        results.sort(key=lambda v: v.created_at, reverse=True)
        
        return results[:query.limit]
    
    # ========================================================================
    # Evaluation Results
    # ========================================================================
    
    def save_evaluation_result(self, result: EvaluationResult) -> EvaluationResult:
        """Save evaluation result."""
        key = f"{result.model_id}_{result.version}"
        
        if key not in self.evaluations:
            self.evaluations[key] = []
        
        self.evaluations[key].append(result)
        self._save_evaluation(result)
        
        logger.info(
            f"[CLM v3 Registry] Saved evaluation for {result.model_id} v{result.version} "
            f"(passed={result.passed}, score={result.promotion_score:.2f}, sharpe={result.sharpe_ratio:.3f})"
        )
        
        return result
    
    def get_evaluation_results(
        self,
        model_id: str,
        version: str,
    ) -> List[EvaluationResult]:
        """Get all evaluation results for a model version."""
        key = f"{model_id}_{version}"
        return self.evaluations.get(key, [])
    
    def get_latest_evaluation(
        self,
        model_id: str,
        version: str,
    ) -> Optional[EvaluationResult]:
        """Get most recent evaluation result."""
        results = self.get_evaluation_results(model_id, version)
        return results[-1] if results else None
    
    # ========================================================================
    # Promotion & Rollback
    # ========================================================================
    
    def promote_model(
        self,
        model_id: str,
        version: str,
        promoted_by: str = "system",
    ) -> bool:
        """
        Promote model to production.
        
        Steps:
        1. Retire current production model
        2. Promote new version to PRODUCTION
        3. Update metadata
        
        Returns:
            True if successful
        """
        # Get model to promote
        model = self.get_model_version(model_id, version)
        if not model:
            logger.error(f"[CLM v3 Registry] Cannot promote {model_id} v{version}: not found")
            return False
        
        # Retire current production model
        current_prod = self.get_production_model(model_id)
        if current_prod:
            current_prod.status = ModelStatus.RETIRED
            current_prod.retired_at = datetime.utcnow()
            self._save_model_metadata(current_prod)
            logger.info(f"[CLM v3 Registry] Retired {model_id} v{current_prod.version}")
        
        # Promote new model
        model.status = ModelStatus.PRODUCTION
        model.promoted_at = datetime.utcnow()
        self._save_model_metadata(model)
        
        logger.info(
            f"[CLM v3 Registry] ✅ Promoted {model_id} v{version} to PRODUCTION "
            f"(by {promoted_by})"
        )
        
        return True
    
    def rollback_to_version(
        self,
        model_id: str,
        target_version: str,
        reason: str = "manual",
    ) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            model_id: Model ID
            target_version: Version to rollback to
            reason: Rollback reason
        
        Returns:
            True if successful
        """
        # Get target model
        target_model = self.get_model_version(model_id, target_version)
        if not target_model:
            logger.error(f"[CLM v3 Registry] Cannot rollback: {model_id} v{target_version} not found")
            return False
        
        # Must be RETIRED or PRODUCTION
        if target_model.status not in [ModelStatus.RETIRED, ModelStatus.PRODUCTION]:
            logger.error(
                f"[CLM v3 Registry] Cannot rollback to {target_model.status} model "
                f"(must be RETIRED or PRODUCTION)"
            )
            return False
        
        # Retire current production
        current_prod = self.get_production_model(model_id)
        if current_prod and current_prod.version != target_version:
            current_prod.status = ModelStatus.RETIRED
            current_prod.retired_at = datetime.utcnow()
            self._save_model_metadata(current_prod)
        
        # Restore target model
        target_model.status = ModelStatus.PRODUCTION
        target_model.promoted_at = datetime.utcnow()
        self._save_model_metadata(target_model)
        
        logger.warning(
            f"[CLM v3 Registry] ⚠️ ROLLBACK: {model_id} → v{target_version} (reason: {reason})"
        )
        
        return True
    
    # ========================================================================
    # Persistence (File-based)
    # ========================================================================
    
    def _save_training_job(self, job: TrainingJob):
        """Save training job metadata to JSON."""
        file_path = self.metadata_dir / "training_jobs" / f"{job.id}.json"
        file_path.parent.mkdir(exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(job.dict(), f, indent=2, default=str)
    
    def _save_model_metadata(self, model: ModelVersion):
        """Save model metadata to JSON."""
        file_path = self.metadata_dir / "models" / model.model_id / f"{model.version}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(model.dict(), f, indent=2, default=str)
    
    def _save_evaluation(self, result: EvaluationResult):
        """Save evaluation result to JSON."""
        file_path = (
            self.metadata_dir / "evaluations" / 
            result.model_id / f"{result.version}_{result.id}.json"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(result.dict(), f, indent=2, default=str)
    
    def _load_metadata(self):
        """Load existing metadata from disk."""
        # Load training jobs
        jobs_dir = self.metadata_dir / "training_jobs"
        if jobs_dir.exists():
            for file_path in jobs_dir.glob("*.json"):
                with open(file_path) as f:
                    data = json.load(f)
                    job = TrainingJob(**data)
                    self.training_jobs[job.id] = job
        
        # Load models
        models_dir = self.metadata_dir / "models"
        if models_dir.exists():
            for model_id_dir in models_dir.iterdir():
                if not model_id_dir.is_dir():
                    continue
                model_id = model_id_dir.name
                self.models[model_id] = {}
                
                for file_path in model_id_dir.glob("*.json"):
                    with open(file_path) as f:
                        data = json.load(f)
                        model = ModelVersion(**data)
                        self.models[model_id][model.version] = model
        
        # Load evaluations
        evals_dir = self.metadata_dir / "evaluations"
        if evals_dir.exists():
            for model_id_dir in evals_dir.iterdir():
                if not model_id_dir.is_dir():
                    continue
                
                for file_path in model_id_dir.glob("*.json"):
                    with open(file_path) as f:
                        data = json.load(f)
                        result = EvaluationResult(**data)
                        key = f"{result.model_id}_{result.version}"
                        if key not in self.evaluations:
                            self.evaluations[key] = []
                        self.evaluations[key].append(result)
        
        logger.info(
            f"[CLM v3 Registry] Loaded metadata: "
            f"{len(self.training_jobs)} jobs, "
            f"{sum(len(v) for v in self.models.values())} model versions, "
            f"{sum(len(v) for v in self.evaluations.values())} evaluations"
        )
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        total_versions = sum(len(versions) for versions in self.models.values())
        
        status_counts = {}
        for versions in self.models.values():
            for model in versions.values():
                status = model.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_model_ids": len(self.models),
            "total_versions": total_versions,
            "status_counts": status_counts,
            "training_jobs": {
                "total": len(self.training_jobs),
                "pending": len([j for j in self.training_jobs.values() if j.status == "pending"]),
                "in_progress": len([j for j in self.training_jobs.values() if j.status == "in_progress"]),
                "completed": len([j for j in self.training_jobs.values() if j.status == "completed"]),
                "failed": len([j for j in self.training_jobs.values() if j.status == "failed"]),
            },
            "evaluations_count": sum(len(v) for v in self.evaluations.values()),
        }

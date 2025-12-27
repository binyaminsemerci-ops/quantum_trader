"""
Retraining Orchestrator - Manages automated model retraining workflows.

Provides:
- Full retraining (all models from scratch)
- Partial retraining (specific models only)
- Incremental learning (update existing models)
- Retraining job scheduling and status tracking
- EventBus integration for triggers
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.domains.learning.data_pipeline import HistoricalDataFetcher, FeatureEngineer, train_val_test_split
from backend.domains.learning.model_registry import ModelRegistry, ModelType, ModelStatus, ModelArtifact
from backend.domains.learning.model_training import (
    train_xgboost,
    train_lightgbm,
    train_nhits,
    train_patchtst,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class RetrainingType(str, Enum):
    """Type of retraining to perform."""
    FULL = "full"  # All models
    PARTIAL = "partial"  # Specific models
    INCREMENTAL = "incremental"  # Update existing models


class JobStatus(str, Enum):
    """Retraining job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RetrainingJob:
    """Retraining job configuration and status."""
    
    job_id: str
    retraining_type: RetrainingType
    model_types: List[ModelType]
    trigger_reason: str
    
    # Configuration
    data_start_date: datetime
    data_end_date: datetime
    training_config: Dict
    
    # Status
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Results
    models_trained: int = 0
    models_succeeded: int = 0
    models_failed: int = 0
    trained_model_ids: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.trained_model_ids is None:
            self.trained_model_ids = []


# ============================================================================
# Retraining Orchestrator
# ============================================================================

class RetrainingOrchestrator:
    """
    Orchestrates automated model retraining.
    
    Workflow:
    1. Receive retraining trigger (drift, schedule, manual)
    2. Fetch historical data
    3. Engineer features
    4. Train models (XGBoost, LightGBM, N-HiTS, PatchTST)
    5. Register new models as SHADOW
    6. Publish completion event
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        policy_store: PolicyStore,
        model_registry: ModelRegistry,
        data_fetcher: HistoricalDataFetcher,
        feature_engineer: FeatureEngineer,
    ):
        self.db = db_session
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.registry = model_registry
        self.data_fetcher = data_fetcher
        self.feature_engineer = feature_engineer
        
        self.running_jobs: Dict[str, RetrainingJob] = {}
        
        logger.info("RetrainingOrchestrator initialized")
    
    async def trigger_retraining(
        self,
        retraining_type: RetrainingType,
        model_types: Optional[List[ModelType]] = None,
        trigger_reason: str = "manual",
        days_of_data: int = 90,
    ) -> str:
        """
        Trigger a retraining job.
        
        Args:
            retraining_type: Type of retraining
            model_types: Models to retrain (None = all)
            trigger_reason: Reason for retraining
            days_of_data: Number of days of historical data
            
        Returns:
            job_id
        """
        # Default to all model types
        if model_types is None:
            model_types = [
                ModelType.XGBOOST,
                ModelType.LIGHTGBM,
                ModelType.NHITS,
                ModelType.PATCHTST,
            ]
        
        # Create job
        job_id = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        data_end_date = datetime.utcnow()
        data_start_date = data_end_date - timedelta(days=days_of_data)
        
        training_config = await self._get_training_config()
        
        job = RetrainingJob(
            job_id=job_id,
            retraining_type=retraining_type,
            model_types=model_types,
            trigger_reason=trigger_reason,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            training_config=training_config,
        )
        
        # Store job
        await self._store_job(job)
        
        # Start async
        asyncio.create_task(self._execute_retraining(job))
        
        logger.info(
            f"Triggered retraining job {job_id}: "
            f"type={retraining_type.value}, models={[m.value for m in model_types]}, "
            f"reason={trigger_reason}"
        )
        
        return job_id
    
    async def _execute_retraining(self, job: RetrainingJob):
        """Execute retraining workflow."""
        self.running_jobs[job.job_id] = job
        
        try:
            # Update status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            await self._update_job(job)
            
            # Publish start event
            await self.event_bus.publish("learning.retraining.started", {
                "job_id": job.job_id,
                "retraining_type": job.retraining_type.value,
                "model_types": [m.value for m in job.model_types],
                "trigger_reason": job.trigger_reason,
            })
            
            # Step 1: Fetch data
            logger.info(f"[{job.job_id}] Fetching data: {job.data_start_date} to {job.data_end_date}")
            
            symbols = await self._get_symbols()
            raw_data = await self.data_fetcher.fetch_historical_data(
                symbols=symbols,
                start_date=job.data_start_date,
                end_date=job.data_end_date,
                timeframe="5m",
            )
            
            # Step 2: Engineer features
            logger.info(f"[{job.job_id}] Engineering features for {len(raw_data)} samples")
            
            feature_data = self.feature_engineer.engineer_features(raw_data)
            feature_data = self.feature_engineer.generate_labels(feature_data, label_type="future_return")
            
            # Split data
            train_df, val_df, test_df = train_val_test_split(feature_data)
            
            label_col = "label"  # Fixed: generate_labels creates "label", not "label_future_return"
            # Select all columns except metadata and label
            exclude_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "label"}
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]
            
            X_train = train_df[feature_cols]
            y_train = train_df[label_col]
            X_val = val_df[feature_cols]
            y_val = val_df[label_col]
            
            logger.info(
                f"[{job.job_id}] Data split: "
                f"train={len(X_train)}, val={len(X_val)}, test={len(test_df)}"
            )
            
            # Step 3: Train models
            config = TrainingConfig(**job.training_config)
            
            for model_type in job.model_types:
                try:
                    logger.info(f"[{job.job_id}] Training {model_type.value}...")
                    
                    # Train
                    if model_type == ModelType.XGBOOST:
                        model, metrics = train_xgboost(X_train, y_train, X_val, y_val, config)
                    elif model_type == ModelType.LIGHTGBM:
                        model, metrics = train_lightgbm(X_train, y_train, X_val, y_val, config)
                    elif model_type == ModelType.NHITS:
                        model, metrics = train_nhits(X_train, y_train, X_val, y_val, config)
                    elif model_type == ModelType.PATCHTST:
                        model, metrics = train_patchtst(X_train, y_train, X_val, y_val, config)
                    else:
                        logger.warning(f"Unknown model type: {model_type.value}")
                        continue
                    
                    # Register as SHADOW
                    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    
                    artifact = ModelArtifact(
                        model_id=f"{model_type.value}_v{version}",
                        model_type=model_type,
                        version=version,
                        status=ModelStatus.SHADOW,
                        metrics=metrics,
                        model_object=model,
                        training_config=job.training_config,
                        training_data_range={
                            "start": job.data_start_date.isoformat(),
                            "end": job.data_end_date.isoformat(),
                        },
                        feature_count=len(feature_cols),
                    )
                    
                    model_id = await self.registry.register_model(artifact)
                    
                    job.models_trained += 1
                    job.models_succeeded += 1
                    job.trained_model_ids.append(model_id)
                    
                    logger.info(
                        f"[{job.job_id}] ✅ Trained {model_type.value}: "
                        f"id={model_id}, val_rmse={metrics.get('val_rmse', 0):.4f}"
                    )
                
                except Exception as e:
                    logger.error(f"[{job.job_id}] ❌ Failed to train {model_type.value}: {e}", exc_info=True)
                    job.models_failed += 1
            
            # Complete job
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            await self._update_job(job)
            
            # Publish completion
            await self.event_bus.publish("learning.retraining.completed", {
                "job_id": job.job_id,
                "models_trained": job.models_trained,
                "models_succeeded": job.models_succeeded,
                "models_failed": job.models_failed,
                "trained_model_ids": job.trained_model_ids,
                "duration_seconds": (job.completed_at - job.started_at).total_seconds(),
            })
            
            logger.info(
                f"[{job.job_id}] ✅ Retraining completed: "
                f"{job.models_succeeded}/{job.models_trained} models succeeded"
            )
        
        except Exception as e:
            # Handle failure
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await self._update_job(job)
            
            logger.error(f"[{job.job_id}] ❌ Retraining failed: {e}", exc_info=True)
            
            await self.event_bus.publish("learning.retraining.failed", {
                "job_id": job.job_id,
                "error": str(e),
            })
        
        finally:
            del self.running_jobs[job.job_id]
    
    async def _get_training_config(self) -> Dict:
        """Get training configuration from PolicyStore."""
        # PolicyStore.get() is synchronous, not async
        config = self.policy_store.get("ml_training_config", {})
        
        defaults = {
            "task": "regression",
            "random_state": 42,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.1,
            "xgb_n_estimators": 100,
            "lgb_num_leaves": 31,
            "lgb_learning_rate": 0.1,
            "lgb_n_estimators": 100,
            "dl_epochs": 50,
            "dl_batch_size": 64,
            "dl_learning_rate": 0.001,
        }
        
        return {**defaults, **config}
    
    async def _get_symbols(self) -> List[str]:
        """Get list of symbols to fetch data for."""
        # PolicyStore.get() is synchronous, not async
        symbols = self.policy_store.get("trading_symbols", [])
        
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Fallback
        
        return symbols
    
    async def _store_job(self, job: RetrainingJob):
        """Store retraining job to database."""
        query = text("""
            INSERT INTO retraining_jobs (
                job_id, retraining_type, model_types, trigger_reason,
                data_start_date, data_end_date, training_config,
                status, created_at
            ) VALUES (
                :job_id, :retraining_type, :model_types, :trigger_reason,
                :data_start_date, :data_end_date, :training_config,
                :status, :created_at
            )
        """)
        
        import json
        self.db.execute(query, {
            "job_id": job.job_id,
            "retraining_type": job.retraining_type.value,
            "model_types": json.dumps([m.value for m in job.model_types]),
            "trigger_reason": job.trigger_reason,
            "data_start_date": job.data_start_date,
            "data_end_date": job.data_end_date,
            "training_config": json.dumps(job.training_config),
            "status": job.status.value,
            "created_at": job.created_at,
        })
        
        self.db.commit()
    
    async def _update_job(self, job: RetrainingJob):
        """Update retraining job status."""
        query = text("""
            UPDATE retraining_jobs SET
                status = :status,
                started_at = :started_at,
                completed_at = :completed_at,
                models_trained = :models_trained,
                models_succeeded = :models_succeeded,
                models_failed = :models_failed,
                trained_model_ids = :trained_model_ids,
                error_message = :error_message
            WHERE job_id = :job_id
        """)
        
        import json
        self.db.execute(query, {
            "job_id": job.job_id,
            "status": job.status.value,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "models_trained": job.models_trained,
            "models_succeeded": job.models_succeeded,
            "models_failed": job.models_failed,
            "trained_model_ids": json.dumps(job.trained_model_ids),
            "error_message": job.error_message,
        })
        
        self.db.commit()
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a retraining job."""
        # Check running jobs first
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "models_trained": job.models_trained,
                "models_succeeded": job.models_succeeded,
                "models_failed": job.models_failed,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
        
        # Query database
        query = text("""
            SELECT *
            FROM retraining_jobs
            WHERE job_id = :job_id
        """)
        
        result = self.db.execute(query, {"job_id": job_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return {
            "job_id": row.job_id,
            "status": row.status,
            "retraining_type": row.retraining_type,
            "trigger_reason": row.trigger_reason,
            "models_trained": row.models_trained,
            "models_succeeded": row.models_succeeded,
            "models_failed": row.models_failed,
            "created_at": row.created_at.isoformat(),
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "error_message": row.error_message,
        }


# ============================================================================
# Database Schema
# ============================================================================

def create_retraining_jobs_table(db_session: Session) -> None:
    """Create retraining_jobs table (SQLite compatible, sync)."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS retraining_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id VARCHAR(255) UNIQUE NOT NULL,
            retraining_type VARCHAR(50) NOT NULL,
            model_types TEXT NOT NULL,
            trigger_reason TEXT NOT NULL,
            data_start_date TIMESTAMP NOT NULL,
            data_end_date TIMESTAMP NOT NULL,
            training_config TEXT NOT NULL,
            status VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            models_trained INTEGER DEFAULT 0,
            models_succeeded INTEGER DEFAULT 0,
            models_failed INTEGER DEFAULT 0,
            trained_model_ids TEXT,
            error_message TEXT
        );
    """)
    
    db_session.execute(create_table_sql)
    db_session.commit()
    logger.info("Created retraining_jobs table")

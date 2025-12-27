"""
CLM v3 Models - Data structures for Continuous Learning Manager v3.

Pydantic models for:
- ModelType (XGB/LGBM/NHITS/PATCHTST/RL_V2/RL_V3)
- TrainingJob (training task specification)
- ModelVersion (versioned model artifacts)
- EvaluationResult (backtest/evaluation metrics)
- TriggerReason (why training was initiated)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class ModelType(str, Enum):
    """Supported model types in CLM v3."""
    
    # Classical ML
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    
    # Deep Time Series
    NHITS = "nhits"
    PATCHTST = "patchtst"
    
    # Reinforcement Learning
    RL_V2 = "rl_v2"  # Legacy RL
    RL_V3 = "rl_v3"  # Current RL PPO
    
    # Experimental
    OTHER = "other"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    
    TRAINING = "training"         # Currently being trained
    SHADOW = "shadow"             # Deployed as shadow (0% allocation)
    CANDIDATE = "candidate"       # Passed evaluation, ready for promotion
    PRODUCTION = "production"     # Active in production (100% allocation)
    RETIRED = "retired"           # Decommissioned
    FAILED = "failed"             # Training or evaluation failed


class TriggerReason(str, Enum):
    """Why training was triggered."""
    
    DRIFT_DETECTED = "drift_detected"           # Model drift from DriftDetectionManager
    PERFORMANCE_DEGRADED = "performance_degraded"  # Poor recent performance
    PERIODIC = "periodic"                       # Scheduled periodic retraining
    MANUAL = "manual"                           # Manual trigger via API
    REGIME_CHANGE = "regime_change"             # Market regime shift detected
    DATA_THRESHOLD = "data_threshold"           # Accumulated enough new data
    STRATEGY_EVOLUTION = "strategy_evolution"   # New strategy candidate


# ============================================================================
# Core Models
# ============================================================================

class TrainingJob(BaseModel):
    """
    Training job specification.
    
    Represents a single training task to be executed by CLM v3.
    """
    
    id: UUID = Field(default_factory=uuid4)
    model_type: ModelType
    symbol: Optional[str] = None  # Specific symbol or None for multi-symbol
    timeframe: str = "1h"
    dataset_span_days: int = 90  # How many days of data to train on
    
    trigger_reason: TriggerReason
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    triggered_by: str = "system"  # system, user email, service name
    
    # Training configuration
    training_params: Dict[str, Any] = Field(default_factory=dict)
    feature_config: Optional[Dict[str, Any]] = None
    
    # Strategy evolution (if applicable)
    strategy_candidate_id: Optional[UUID] = None
    parent_model_version: Optional[str] = None
    
    # Status tracking
    status: str = "pending"  # pending, in_progress, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ModelVersion(BaseModel):
    """
    Versioned model artifact with metadata.
    
    Represents a specific trained model version in the registry.
    """
    
    model_id: str  # e.g., "xgboost_btcusdt_1h"
    version: str   # e.g., "v20251204_143022" or "1.0.0"
    model_type: ModelType
    
    # Lifecycle
    status: ModelStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    
    # Relationships
    parent_version: Optional[str] = None  # Previous version
    training_job_id: Optional[UUID] = None
    
    # Artifacts
    model_path: str  # File path or storage key
    model_size_bytes: int
    model_hash: Optional[str] = None  # SHA256 hash for integrity
    
    # Metadata
    training_data_range: Dict[str, str]  # {"start": "2024-01-01", "end": "2024-03-01"}
    feature_count: int
    training_params: Dict[str, Any]
    
    # Performance (at training time)
    train_metrics: Dict[str, float]  # Loss, accuracy, etc.
    validation_metrics: Dict[str, float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class EvaluationResult(BaseModel):
    """
    Model evaluation/backtest result.
    
    Used to decide if model should be promoted to CANDIDATE/PRODUCTION.
    """
    
    id: UUID = Field(default_factory=uuid4)
    model_id: str
    version: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Evaluation type
    evaluation_type: str = "backtest"  # backtest, shadow_test, walk_forward
    evaluation_period_days: int
    
    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    
    # Risk metrics
    risk_adjusted_return: float
    calmar_ratio: float
    
    # Decision
    passed: bool  # Did it pass promotion criteria?
    promotion_score: float  # 0-100 score
    failure_reason: Optional[str] = None
    
    # Thresholds used
    min_sharpe: float = 1.0
    min_win_rate: float = 0.52
    min_profit_factor: float = 1.3
    max_drawdown_threshold: float = 0.15
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# ============================================================================
# Registry Query Models
# ============================================================================

class ModelQuery(BaseModel):
    """Query filter for model registry searches."""
    
    model_type: Optional[ModelType] = None
    status: Optional[ModelStatus] = None
    min_sharpe: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = 50


class PromotionRequest(BaseModel):
    """Request to promote a model to production."""
    
    model_id: str
    version: str
    promoted_by: str  # User or service name
    reason: str
    force: bool = False  # Skip safety checks


class RollbackRequest(BaseModel):
    """Request to rollback to a previous model version."""
    
    model_id: str
    target_version: str
    reason: str
    rollback_by: str


# ============================================================================
# Event Payloads (for EventBus)
# ============================================================================

class TrainingJobCreatedEvent(BaseModel):
    """Event: New training job created."""
    
    job_id: UUID
    model_type: ModelType
    trigger_reason: TriggerReason
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelTrainedEvent(BaseModel):
    """Event: Model training completed."""
    
    job_id: UUID
    model_id: str
    version: str
    model_type: ModelType
    success: bool
    train_metrics: Dict[str, float]
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class ModelEvaluatedEvent(BaseModel):
    """Event: Model evaluation completed."""
    
    model_id: str
    version: str
    passed: bool
    promotion_score: float
    metrics: Dict[str, float]
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelPromotedEvent(BaseModel):
    """Event: Model promoted to production."""
    
    model_id: str
    version: str
    previous_version: Optional[str]
    promoted_by: str
    promoted_at: datetime = Field(default_factory=datetime.utcnow)


class ModelRollbackEvent(BaseModel):
    """Event: Model rolled back to previous version."""
    
    model_id: str
    from_version: str
    to_version: str
    reason: str
    rollback_by: str
    rollback_at: datetime = Field(default_factory=datetime.utcnow)


class StrategyCandidateCreatedEvent(BaseModel):
    """Event: New strategy candidate created by Evolution Engine."""
    
    candidate_id: UUID
    base_strategy: str
    mutation_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

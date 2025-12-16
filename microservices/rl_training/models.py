"""
Data models for RL Training Service

Events, training jobs, model artifacts, drift alerts.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ModelType(str, Enum):
    """Types of models managed by this service"""
    RL_PPO = "rl_ppo"
    RL_SIZING = "rl_sizing"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NHITS = "nhits"
    PATCHTST = "patchtst"


class ModelStatus(str, Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class TrainingTrigger(str, Enum):
    """Why training was triggered"""
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DECAY = "performance_decay"
    MANUAL = "manual"
    REGIME_SHIFT = "regime_shift"


class DriftSeverity(str, Enum):
    """Drift severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


# ============================================================================
# EVENTS IN (subscribed by this service)
# ============================================================================

class PerformanceMetricsUpdatedEvent(BaseModel):
    """Published by AI Engine / Portfolio Intelligence"""
    model_name: str
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    timestamp: str


class DataDriftSignalEvent(BaseModel):
    """Published by market data service or internal drift detector"""
    drift_type: str  # "feature_drift", "concept_drift"
    severity: DriftSeverity
    affected_features: List[str]
    psi_score: Optional[float] = None
    timestamp: str


class ManualRetrainRequestEvent(BaseModel):
    """Manual retrain request from admin API"""
    model_type: ModelType
    reason: str
    requested_by: str
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    timestamp: str


# ============================================================================
# EVENTS OUT (published by this service)
# ============================================================================

class ModelTrainingStartedEvent(BaseModel):
    """Training job started"""
    job_id: str
    model_type: ModelType
    trigger: TrainingTrigger
    reason: str
    started_at: str


class ModelTrainingCompletedEvent(BaseModel):
    """Training job completed"""
    job_id: str
    model_type: ModelType
    model_version: str
    status: str  # "success" or "failed"
    metrics: Dict[str, float]
    training_duration_seconds: float
    completed_at: str


class ModelPromotedEvent(BaseModel):
    """Model promoted from shadow to active"""
    model_type: ModelType
    old_version: Optional[str]
    new_version: str
    promotion_reason: str
    improvement_pct: float
    promoted_at: str


class ModelRegressedEvent(BaseModel):
    """Model performance regressed, rolled back"""
    model_type: ModelType
    failed_version: str
    rolled_back_to: str
    regression_reason: str
    regressed_at: str


class DriftDetectedEvent(BaseModel):
    """Data drift detected"""
    drift_type: str
    severity: DriftSeverity
    affected_models: List[str]
    psi_score: float
    recommendation: str  # "monitor", "retrain_scheduled", "retrain_urgent"
    detected_at: str


# ============================================================================
# API MODELS
# ============================================================================

class TrainingJobInfo(BaseModel):
    """Training job details"""
    job_id: str
    model_type: ModelType
    status: str
    trigger: TrainingTrigger
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class ModelVersionInfo(BaseModel):
    """Model version metadata"""
    model_type: ModelType
    version: str
    status: ModelStatus
    trained_at: str
    metrics: Dict[str, float]
    is_active: bool


class ShadowModelStatus(BaseModel):
    """Shadow model status"""
    model_name: str
    model_type: ModelType
    version: str
    shadow_since: str
    num_predictions: int
    performance_vs_champion: Dict[str, float]  # {"sharpe_diff": +0.05, "winrate_diff": +0.02}
    ready_for_promotion: bool


class DriftMetrics(BaseModel):
    """Drift detection metrics"""
    feature_name: str
    psi_score: float
    severity: DriftSeverity
    last_checked: str


class ServiceHealth(BaseModel):
    """Service health status"""
    service: str
    version: str
    status: str
    uptime_seconds: float
    components: List[Dict[str, Any]]
    last_training_job: Optional[TrainingJobInfo] = None


class TrainingHistoryEntry(BaseModel):
    """Training history record"""
    job_id: str
    model_type: ModelType
    trigger: TrainingTrigger
    started_at: str
    completed_at: Optional[str] = None
    status: str
    metrics: Optional[Dict[str, float]] = None


# ============================================================================
# INTERNAL MODELS
# ============================================================================

class TrainingConfig(BaseModel):
    """Configuration for a training job"""
    model_type: ModelType
    data_lookback_days: int = 90
    min_samples: int = 100
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    features: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Model evaluation result"""
    model_version: str
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    is_better_than_baseline: bool
    confidence_interval: Optional[Dict[str, float]] = None

"""
Data models for Continuous Learning Manager.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from enum import Enum


class ModelStage(Enum):
    """Lifecycle stages for ML models."""
    TRAINING = "TRAINING"
    SHADOW = "SHADOW"
    LIVE = "LIVE"
    RETIRED = "RETIRED"


@dataclass
class ModelVersion:
    """Represents a specific version of an ML model."""
    model_name: str
    version: str
    
    # Model details
    stage: ModelStage = ModelStage.SHADOW
    model_type: str = "lstm"  # lstm, transformer, etc.
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Training info
    trained_on_samples: int = 0
    training_time_seconds: float = 0.0
    hyperparameters: Dict = field(default_factory=dict)
    
    # Lifecycle timestamps
    created_at: datetime = field(default_factory=datetime.now)
    promoted_to_live_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage.value,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "trained_on_samples": self.trained_on_samples,
            "training_time_seconds": self.training_time_seconds,
            "hyperparameters": self.hyperparameters,
            "created_at": self.created_at.isoformat(),
            "promoted_to_live_at": self.promoted_to_live_at.isoformat() if self.promoted_to_live_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
        }


@dataclass
class ShadowEvaluationResult:
    """Results from shadow mode evaluation."""
    model_name: str
    shadow_version: str
    live_version: str
    
    # Comparative metrics
    shadow_accuracy: float
    live_accuracy: float
    accuracy_improvement: float
    
    shadow_f1: float
    live_f1: float
    f1_improvement: float
    
    # Evaluation details
    samples_evaluated: int
    evaluation_period_hours: float
    
    # Recommendation
    should_promote: bool
    confidence: float  # 0-1
    reason: str
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "shadow_version": self.shadow_version,
            "live_version": self.live_version,
            "shadow_accuracy": self.shadow_accuracy,
            "live_accuracy": self.live_accuracy,
            "accuracy_improvement": self.accuracy_improvement,
            "shadow_f1": self.shadow_f1,
            "live_f1": self.live_f1,
            "f1_improvement": self.f1_improvement,
            "samples_evaluated": self.samples_evaluated,
            "evaluation_period_hours": self.evaluation_period_hours,
            "should_promote": self.should_promote,
            "confidence": self.confidence,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RetrainingConfig:
    """Configuration for model retraining."""
    model_name: str
    
    # Retraining schedule
    retrain_interval_hours: float = 24.0  # Daily by default
    min_new_samples: int = 1000  # Minimum new samples to trigger retraining
    
    # Shadow evaluation
    shadow_evaluation_hours: float = 24.0  # Evaluate for 24h before promotion
    min_improvement_threshold: float = 0.02  # 2% improvement required
    
    # Model parameters
    hyperparameters: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "retrain_interval_hours": self.retrain_interval_hours,
            "min_new_samples": self.min_new_samples,
            "shadow_evaluation_hours": self.shadow_evaluation_hours,
            "min_improvement_threshold": self.min_improvement_threshold,
            "hyperparameters": self.hyperparameters,
        }

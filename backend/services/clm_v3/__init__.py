"""
CLM v3 - Continuous Learning Manager v3

Enhanced retraining orchestration with:
- Safe model promotion/rollback workflows
- Strategy Evolution Engine integration
- Multi-model type support (ML + Deep Learning + RL)
- EventBus-driven triggers
- Model Registry with versioning

Author: Quantum Trader AI Team
Version: 3.0
Date: December 4, 2025
"""

from backend.services.clm_v3.models import (
    ModelType,
    ModelStatus,
    TrainingJob,
    ModelVersion,
    EvaluationResult,
    TriggerReason,
)

from backend.services.clm_v3.orchestrator import ClmOrchestrator
from backend.services.clm_v3.storage import ModelRegistryV3
from backend.services.clm_v3.scheduler import TrainingScheduler
from backend.services.clm_v3.strategies import StrategyEvolutionEngine, StrategyCandidate

__all__ = [
    "ModelType",
    "ModelStatus",
    "TrainingJob",
    "ModelVersion",
    "EvaluationResult",
    "TriggerReason",
    "ClmOrchestrator",
    "ModelRegistryV3",
    "TrainingScheduler",
    "StrategyEvolutionEngine",
    "StrategyCandidate",
]

__version__ = "3.0.0"

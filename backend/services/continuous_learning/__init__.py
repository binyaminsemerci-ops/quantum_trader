"""
Continuous Learning Manager (CLM) for Quantum Trader AI OS.

Handles model retraining, shadow evaluation, and automatic promotion.
"""

from .models import (
    ModelVersion,
    ShadowEvaluationResult,
    RetrainingConfig,
    ModelStage,
)
from .manager import ContinuousLearningManager
from .evaluator import ShadowEvaluator

__all__ = [
    "ContinuousLearningManager",
    "ShadowEvaluator",
    "ModelVersion",
    "ShadowEvaluationResult",
    "RetrainingConfig",
    "ModelStage",
]

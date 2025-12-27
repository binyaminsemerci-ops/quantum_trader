"""
Continuous Learning System

Wraps RetrainingOrchestrator for backward compatibility
"""
from backend.services.retraining_orchestrator import RetrainingOrchestrator

# Alias for compatibility
ContinuousLearningSystem = RetrainingOrchestrator


def get_continuous_learning():
    """Get continuous learning orchestrator"""
    return RetrainingOrchestrator()

"""Self-Evolving Strategy Architect (SESA) for Quantum Trader v5.

This package provides automated strategy evolution capabilities:
- SESA Engine: Main orchestrator for strategy evolution
- Mutation Operators: Generate strategy variations
- Evaluation Engine: Evaluate strategies via backtesting
- Selection Engine: Select top performers for deployment
"""

from backend.sesa.sesa_engine import SESAEngine, EvolutionConfig, EvolutionResult
from backend.sesa.mutation_operators import (
    MutationOperators,
    StrategyConfig,
    MutatedStrategy,
    MutationType,
)
from backend.sesa.evaluation_engine import (
    EvaluationEngine,
    EvaluationResult,
    PerformanceMetrics,
)
from backend.sesa.selection_engine import (
    SelectionEngine,
    SelectionResult,
    CandidateStatus,
)

__all__ = [
    "SESAEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "MutationOperators",
    "StrategyConfig",
    "MutatedStrategy",
    "MutationType",
    "EvaluationEngine",
    "EvaluationResult",
    "PerformanceMetrics",
    "SelectionEngine",
    "SelectionResult",
    "CandidateStatus",
]

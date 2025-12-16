"""
CLM (Continuous Learning Manager) - Real Implementations

This package contains production-ready implementations of CLM components:
- RealDataClient: BinanceDataFetcher integration
- RealModelTrainer: Actual model training for all model types
- RealModelEvaluator: Comprehensive metric calculations
- RealShadowTester: Live parallel testing framework
- RealModelRegistry: PostgreSQL-based model versioning
"""

from .data_client import RealDataClient
from .model_trainer import RealModelTrainer
from .model_evaluator import RealModelEvaluator
from .shadow_tester import RealShadowTester
from .model_registry import RealModelRegistry

__all__ = [
    "RealDataClient",
    "RealModelTrainer",
    "RealModelEvaluator",
    "RealShadowTester",
    "RealModelRegistry",
]

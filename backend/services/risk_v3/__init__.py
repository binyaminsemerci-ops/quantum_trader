"""
Global Risk Engine v3 for Quantum Trader v2.0

EPIC-RISK3-001: Multi-exchange, multi-symbol, multi-strategy risk architecture

Components:
- Exposure Matrix: Cross-symbol/exchange/strategy correlation and exposure
- VaR/ES: Value at Risk and Expected Shortfall computation
- Systemic Risk: Liquidity stress, correlation spikes, contagion detection
- Risk Orchestrator: Central risk evaluation engine with ESS v3 and Federation AI integration
- Adapters: Integration with Portfolio, Exchange, PolicyStore, Federation AI

Event Flow:
- risk.global_snapshot
- risk.var_es_updated
- risk.systemic_alert
- risk.exposure_matrix_updated
- risk.threshold_breach
"""

from .models import (
    RiskSnapshot,
    ExposureMatrix,
    VaRResult,
    ESResult,
    SystemicRiskSignal,
    GlobalRiskSignal,
    RiskLevel,
    SystemicRiskType,
)

from .orchestrator import RiskOrchestrator
from .exposure_matrix import ExposureMatrixEngine
from .var_es import VaRESEngine
from .systemic import SystemicRiskDetector

__all__ = [
    "RiskSnapshot",
    "ExposureMatrix",
    "VaRResult",
    "ESResult",
    "SystemicRiskSignal",
    "GlobalRiskSignal",
    "RiskLevel",
    "SystemicRiskType",
    "RiskOrchestrator",
    "ExposureMatrixEngine",
    "VaRESEngine",
    "SystemicRiskDetector",
]

__version__ = "3.0.0"

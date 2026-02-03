"""
Risk Guard Module - Fail-Closed Risk Management

Components:
- RiskGuard: Global risk gates (daily loss, drawdown, consecutive losses, spread/volatility spikes)
- ATRPositionSizer: Dynamic position sizing based on ATR and regime
- RobustExitEngine: Continuous exit monitoring with reduceOnly plan emission

All components are fail-closed: missing data → BLOCK/CLOSE
"""

from .risk_guard import RiskGuard, RiskGuardConfig
from .atr_sizer import ATRPositionSizer, ATRSizerConfig
from .robust_exit_engine import RobustExitEngine, ExitEngineConfig

__all__ = [
    'RiskGuard',
    'RiskGuardConfig',
    'ATRPositionSizer',
    'ATRSizerConfig',
    'RobustExitEngine',
    'ExitEngineConfig'
]

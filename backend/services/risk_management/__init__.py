"""Risk Management Module - ATR-based profit-optimized trading."""

from backend.services.risk_management.trade_opportunity_filter import (
    ConsensusType,
    FilterResult,
    MarketConditions,
    SignalQuality,
    TradeOpportunityFilter,
)
from backend.services.risk_management.risk_manager import (
    PositionSize,
    RiskManager,
)
from backend.services.risk_management.exit_policy_engine import (
    ExitDecision,
    ExitLevels,
    ExitPolicyEngine,
    ExitSignal,
)
from backend.services.risk_management.global_risk_controller import (
    GlobalRiskController,
    PositionInfo,
    RiskCheckResult,
    RiskStatus,
    TradeRecord,
)
from backend.services.risk_management.trade_lifecycle_manager import (
    ManagedTrade,
    TradeDecision,
    TradeLifecycleManager,
    TradeState,
)

__all__ = [
    # Trade Opportunity Filter
    "ConsensusType",
    "FilterResult",
    "MarketConditions",
    "SignalQuality",
    "TradeOpportunityFilter",
    # Risk Manager
    "PositionSize",
    "RiskManager",
    # Exit Policy Engine
    "ExitDecision",
    "ExitLevels",
    "ExitPolicyEngine",
    "ExitSignal",
    # Global Risk Controller
    "GlobalRiskController",
    "PositionInfo",
    "RiskCheckResult",
    "RiskStatus",
    "TradeRecord",
    # Trade Lifecycle Manager
    "ManagedTrade",
    "TradeDecision",
    "TradeLifecycleManager",
    "TradeState",
]

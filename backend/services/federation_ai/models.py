"""
Federation AI v3 - Decision Models
===================================

Pydantic models for Federation decisions, events, and state.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from uuid import uuid4


# ============================================================================
# ENUMS
# ============================================================================

class TradingMode(str, Enum):
    """Global trading mode"""
    LIVE = "LIVE"           # Active trading
    SHADOW = "SHADOW"       # Observe only, no execution
    PAUSED = "PAUSED"       # Temporarily disabled
    EMERGENCY = "EMERGENCY"  # Emergency stop


class CapitalProfile(str, Enum):
    """Capital risk profiles"""
    MICRO = "MICRO"           # 0.1-0.5% per trade, ultra-safe
    LOW = "LOW"               # 0.5-1% per trade, conservative
    NORMAL = "NORMAL"         # 1-2% per trade, standard
    AGGRESSIVE = "AGGRESSIVE"  # 2-5% per trade, high risk
    

class DecisionPriority(int, Enum):
    """Decision priority levels"""
    CRITICAL = 1   # Immediate execution (override, freeze)
    HIGH = 2       # Urgent (risk adjustments)
    NORMAL = 3     # Standard (strategy allocation)
    LOW = 4        # Background (research tasks)


class DecisionType(str, Enum):
    """Federation decision types"""
    MODE_CHANGE = "MODE_CHANGE"
    CAPITAL_PROFILE = "CAPITAL_PROFILE"
    RISK_ADJUSTMENT = "RISK_ADJUSTMENT"
    STRATEGY_ALLOCATION = "STRATEGY_ALLOCATION"
    SYMBOL_UNIVERSE = "SYMBOL_UNIVERSE"
    ESS_POLICY = "ESS_POLICY"
    CASHFLOW = "CASHFLOW"
    RESEARCH_TASK = "RESEARCH_TASK"
    OVERRIDE = "OVERRIDE"
    FREEZE = "FREEZE"


# ============================================================================
# BASE MODELS
# ============================================================================

class FederationDecision(BaseModel):
    """Base class for all Federation decisions"""
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_type: DecisionType
    role_source: str  # "ceo", "cio", "cro", "cfo", "researcher", "supervisor"
    priority: DecisionPriority
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None  # Link related decisions
    reason: str
    payload: Dict[str, Any]
    
    class Config:
        use_enum_values = True


class CapitalProfileDecision(BaseModel):
    """AI-CEO: Capital profile change"""
    profile: CapitalProfile
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    max_risk_per_trade_pct: float  # 0.01 = 1%
    max_daily_risk_pct: float
    max_positions: int
    
    class Config:
        use_enum_values = True


class TradingModeDecision(BaseModel):
    """AI-CEO: Trading mode change"""
    mode: TradingMode
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_minutes: Optional[int] = None  # For temporary pauses
    
    class Config:
        use_enum_values = True


class RiskAdjustmentDecision(BaseModel):
    """AI-CRO: Risk limit adjustments"""
    max_leverage: float
    max_position_size_usd: float
    max_drawdown_pct: float
    max_exposure_pct: float  # Of total capital
    stop_loss_multiplier: float  # Tighten (>1.0) or loosen (<1.0)
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ESSPolicyDecision(BaseModel):
    """AI-CRO: ESS threshold adjustments"""
    caution_threshold_pct: float   # e.g., 0.03 = 3% DD
    warning_threshold_pct: float   # e.g., 0.05 = 5% DD
    critical_threshold_pct: float  # e.g., 0.08 = 8% DD
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StrategyAllocationDecision(BaseModel):
    """AI-CIO: Strategy mix and model weights"""
    model_weights: Dict[str, float]  # {"xgboost": 0.4, "lightgbm": 0.3, ...}
    active_strategies: List[str]     # ["trend_following", "mean_reversion"]
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SymbolUniverseDecision(BaseModel):
    """AI-CIO: Symbol selection"""
    active_symbols: List[str]  # ["BTCUSDT", "ETHUSDT"]
    excluded_symbols: List[str]  # Temporarily banned
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CashflowDecision(BaseModel):
    """AI-CFO: Capital allocation"""
    profit_lock_pct: float       # Lock profits (withdraw)
    reinvest_pct: float          # Reinvest into trading
    reserve_buffer_pct: float    # Cash buffer
    reason: str
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchTaskDecision(BaseModel):
    """AI-Researcher: Research proposal"""
    task_type: str  # "hyperparameter_tuning", "new_feature", "strategy_backtest"
    description: str
    priority: str  # "high", "normal", "low"
    estimated_duration_hours: float
    resources_needed: List[str]


class OverrideDecision(BaseModel):
    """AI-Supervisor: Override another decision"""
    overridden_decision_id: str
    overridden_role: str
    reason: str
    new_directive: Optional[Dict[str, Any]] = None


class FreezeDecision(BaseModel):
    """AI-Supervisor: Emergency freeze trading"""
    reason: str
    duration_minutes: Optional[int] = None  # None = indefinite
    affected_subsystems: List[str]  # ["execution", "ai_engine", "all"]
    severity: str  # "warning", "critical", "emergency"


# ============================================================================
# INPUT MODELS (Events from other systems)
# ============================================================================

class PortfolioSnapshot(BaseModel):
    """Portfolio state from Portfolio Intelligence"""
    timestamp: datetime
    total_equity: float
    unrealized_pnl: float
    realized_pnl_today: float
    drawdown_pct: float
    max_drawdown_pct: float
    num_positions: int
    total_exposure_usd: float
    cash_balance: float
    win_rate_today: float
    sharpe_ratio_7d: Optional[float] = None


class SystemHealthSnapshot(BaseModel):
    """System health from Health Monitor"""
    timestamp: datetime
    overall_status: str  # "OPTIMAL", "HEALTHY", "DEGRADED", "CRITICAL"
    subsystem_scores: Dict[str, float]  # {"ai_engine": 95.0, "execution": 88.0}
    active_alerts: List[str]
    ess_state: str  # "NOMINAL", "CAUTION", "WARNING", "CRITICAL"


class ModelPerformance(BaseModel):
    """Model performance from Model Registry"""
    model_name: str
    accuracy_7d: float
    sharpe_ratio_7d: float
    win_rate_7d: float
    bias_detected: bool
    shadow_vote_agreement_pct: float

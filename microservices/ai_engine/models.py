"""
AI Engine Service - Pydantic Models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# ========================================================================
# ENUMS
# ========================================================================

class SignalAction(str, Enum):
    """Trading signal action."""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class MarketRegime(str, Enum):
    """Market regime classification."""
    HIGH_VOL_TRENDING = "high_vol_trending"
    LOW_VOL_TRENDING = "low_vol_trending"
    HIGH_VOL_RANGING = "high_vol_ranging"
    LOW_VOL_RANGING = "low_vol_ranging"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


class StrategyID(str, Enum):
    """Strategy identifiers."""
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    MEAN_REVERT = "mean_revert"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    DEFAULT = "default"


# ========================================================================
# EVENT MODELS (Incoming)
# ========================================================================

class MarketTickEvent(BaseModel):
    """Market tick event from marketdata-service."""
    symbol: str
    price: float
    volume: float = 0.0
    timestamp: str


class MarketKlineEvent(BaseModel):
    """Market kline/candle event."""
    symbol: str
    timeframe: str = "5m"
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str


class TradeClosedEvent(BaseModel):
    """Trade closed event from execution-service (for continuous learning)."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_percent: float
    model: str
    confidence: float
    strategy: Optional[str] = None
    timestamp: str


class PolicyUpdatedEvent(BaseModel):
    """Policy updated event from risk-safety-service."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: str


# ========================================================================
# EVENT MODELS (Outgoing)
# ========================================================================

class AISignalGeneratedEvent(BaseModel):
    """AI signal generated (intermediate event)."""
    symbol: str
    action: SignalAction
    confidence: float
    ensemble_confidence: float
    model_votes: Dict[str, str]  # {"xgb": "buy", "lgbm": "buy", ...}
    consensus: int  # Number of models in agreement
    timestamp: str


class StrategySelectedEvent(BaseModel):
    """Strategy selected by Meta-Strategy Selector."""
    symbol: str
    strategy_id: StrategyID
    strategy_name: str
    regime: MarketRegime
    confidence: float
    reasoning: str
    is_exploration: bool
    q_values: Dict[str, float]
    timestamp: str


class SizingDecidedEvent(BaseModel):
    """Position sizing decided by RL Agent."""
    symbol: str
    position_size_usd: float
    leverage: float
    risk_pct: float
    confidence: float
    reasoning: str
    tp_percent: float
    sl_percent: float
    partial_tp_enabled: bool
    timestamp: str


class AIDecisionMadeEvent(BaseModel):
    """
    Final AI decision (trade intent).
    
    This is the main output event consumed by execution-service.
    """
    symbol: str
    side: SignalAction
    confidence: float
    
    # Entry parameters
    entry_price: Optional[float] = None  # Market price if None
    quantity: float = 0.0  # USD value
    leverage: int = 1
    
    # TP/SL parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Metadata
    model: str = "ensemble"
    ensemble_confidence: float = 0.0
    strategy: str = "default"
    meta_strategy: Optional[str] = None
    regime: Optional[MarketRegime] = None
    position_size_usd: float = 0.0
    
    timestamp: str


# ========================================================================
# API REQUEST/RESPONSE MODELS
# ========================================================================

class SignalRequest(BaseModel):
    """Manual signal generation request."""
    symbol: str
    timeframe: str = "5m"
    include_reasoning: bool = False


class SignalResponse(BaseModel):
    """Signal generation response."""
    symbol: str
    action: SignalAction
    confidence: float
    ensemble_confidence: float
    strategy: str
    regime: MarketRegime
    position_size_usd: float
    leverage: int
    reasoning: Optional[str] = None
    timestamp: str


class ModelInfo(BaseModel):
    """AI model information."""
    model_name: str
    model_type: str  # "xgb", "lgbm", "nhits", "patchtst"
    loaded: bool
    version: Optional[str] = None
    last_inference: Optional[str] = None


class ComponentHealth(BaseModel):
    """Component health status."""
    healthy: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ServiceHealth(BaseModel):
    """Service health response."""
    service: str
    healthy: bool
    running: bool
    components: Dict[str, ComponentHealth]
    models_loaded: int
    signals_generated_total: int
    timestamp: str


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: str


class EnsembleMetrics(BaseModel):
    """Ensemble performance metrics."""
    total_signals: int
    consensus_rate: float  # % of signals with 3/4 agreement
    avg_confidence: float
    model_agreement: Dict[str, float]  # Agreement rate per model
    last_updated: str


class MetaStrategyMetrics(BaseModel):
    """Meta-strategy performance metrics."""
    total_decisions: int
    exploration_rate: float
    top_strategies: List[Dict[str, Any]]  # Top 3 strategies by Q-value
    avg_q_value: float
    last_updated: str


class RLSizingMetrics(BaseModel):
    """RL position sizing metrics."""
    total_decisions: int
    exploration_rate: float
    avg_position_size_usd: float
    avg_leverage: float
    avg_risk_pct: float
    last_updated: str

"""
Risk v3 Models - Pydantic models for Global Risk Engine

EPIC-RISK3-001: Core data models for risk management
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SystemicRiskType(str, Enum):
    """Types of systemic risk events"""
    LIQUIDITY_STRESS = "liquidity_stress"
    CORRELATION_SPIKE = "correlation_spike"
    MULTI_EXCHANGE_FAILURE = "multi_exchange_failure"
    VOLATILITY_REGIME_SHIFT = "volatility_regime_shift"
    CASCADING_RISK = "cascading_risk"
    CONCENTRATION_RISK = "concentration_risk"


class PositionExposure(BaseModel):
    """Single position exposure details"""
    symbol: str
    exchange: str
    strategy: str
    quantity: float
    notional_usd: float
    leverage: float
    unrealized_pnl: float
    entry_price: float
    current_price: float
    risk_pct: float  # % of account
    

class RiskSnapshot(BaseModel):
    """Complete risk snapshot at a point in time
    
    Aggregates all positions, exposures, and account state for risk evaluation
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Position-level data
    positions: List[PositionExposure] = Field(default_factory=list)
    
    # Symbol-level exposure
    symbol_exposure: Dict[str, float] = Field(default_factory=dict)  # symbol -> notional USD
    symbol_leverage: Dict[str, float] = Field(default_factory=dict)  # symbol -> avg leverage
    
    # Exchange-level exposure
    exchange_exposure: Dict[str, float] = Field(default_factory=dict)  # exchange -> notional USD
    exchange_position_count: Dict[str, int] = Field(default_factory=dict)
    
    # Strategy-level exposure
    strategy_exposure: Dict[str, float] = Field(default_factory=dict)  # strategy -> notional USD
    strategy_position_count: Dict[str, int] = Field(default_factory=dict)
    
    # Account state
    account_balance: float
    total_equity: float
    total_notional: float  # Sum of all position notionals
    total_unrealized_pnl: float
    
    # Risk metrics
    total_leverage: float  # total_notional / total_equity
    drawdown_pct: float  # Current drawdown from peak
    daily_pnl: float
    weekly_pnl: float
    
    # Market state
    volatility_cluster: Optional[str] = None  # "low", "medium", "high", "extreme"
    regime: Optional[str] = None  # From regime detector
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-04T10:00:00Z",
                "account_balance": 10000.0,
                "total_equity": 10500.0,
                "total_notional": 25000.0,
                "total_leverage": 2.38,
                "drawdown_pct": 0.05,
                "symbol_exposure": {"BTCUSDT": 15000.0, "ETHUSDT": 10000.0},
                "exchange_exposure": {"binance": 25000.0}
            }
        }


class CorrelationMatrix(BaseModel):
    """Correlation matrix between symbols/strategies"""
    symbols: List[str]
    matrix: List[List[float]]  # NxN correlation matrix
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    lookback_periods: int = 30
    method: str = "pearson"  # pearson, spearman, kendall


class ExposureMatrix(BaseModel):
    """Multi-dimensional exposure analysis
    
    Tracks correlation, concentration, and beta-weighted exposure across:
    - Symbols
    - Exchanges  
    - Strategies
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Correlation analysis
    correlation_matrix: Optional[CorrelationMatrix] = None
    avg_correlation: float = 0.0  # Average pairwise correlation
    max_correlation: float = 0.0  # Highest pairwise correlation
    
    # Beta weights (placeholder for future enhancement)
    beta_weights: Dict[str, float] = Field(default_factory=dict)  # symbol -> beta vs benchmark
    
    # Normalized exposure (0-1 scale)
    normalized_symbol_exposure: Dict[str, float] = Field(default_factory=dict)
    normalized_exchange_exposure: Dict[str, float] = Field(default_factory=dict)
    normalized_strategy_exposure: Dict[str, float] = Field(default_factory=dict)
    
    # Risk hotspots (top concentrated exposures)
    risk_hotspots: List[Dict[str, Any]] = Field(default_factory=list)
    # Example: [{"type": "symbol", "name": "BTCUSDT", "exposure_pct": 0.65, "risk_score": 0.8}]
    
    # Concentration metrics
    symbol_concentration_hhi: float = 0.0  # Herfindahl-Hirschman Index
    exchange_concentration_hhi: float = 0.0
    strategy_concentration_hhi: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-04T10:00:00Z",
                "avg_correlation": 0.45,
                "max_correlation": 0.82,
                "symbol_concentration_hhi": 0.35,
                "risk_hotspots": [
                    {"type": "symbol", "name": "BTCUSDT", "exposure_pct": 0.60, "risk_score": 0.75}
                ]
            }
        }


class VaRResult(BaseModel):
    """Value at Risk calculation result"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # VaR at different confidence levels
    var_95: float  # 95% confidence level
    var_99: float  # 99% confidence level
    
    # Methodology
    method: str = "delta_normal"  # delta_normal, historical, monte_carlo
    lookback_periods: int = 30
    time_horizon_hours: int = 24
    
    # Thresholds
    threshold_95: float  # Max acceptable VaR at 95%
    threshold_99: float  # Max acceptable VaR at 99%
    
    # Pass/fail
    pass_95: bool
    pass_99: bool
    
    # Additional context
    portfolio_volatility: float  # Annualized volatility
    note: Optional[str] = None


class ESResult(BaseModel):
    """Expected Shortfall (Conditional VaR) calculation result"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Expected Shortfall at 97.5% confidence
    es_975: float
    
    # Methodology
    method: str = "historical"  # historical, parametric
    lookback_periods: int = 30
    
    # Threshold
    threshold_975: float
    pass_975: bool
    
    # Additional metrics
    worst_case_loss: float  # Maximum historical loss in tail
    tail_events_count: int  # Number of events in tail
    note: Optional[str] = None


class SystemicRiskSignal(BaseModel):
    """Systemic risk detection signal
    
    Identifies market-wide or portfolio-wide risk conditions that could
    trigger cascading failures or significant losses
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    level: RiskLevel  # INFO, WARNING, CRITICAL
    risk_type: SystemicRiskType
    
    # Detection details
    description: str
    severity_score: float = Field(ge=0.0, le=1.0)  # 0 = no risk, 1 = extreme risk
    
    # Contributing factors
    factors: Dict[str, Any] = Field(default_factory=dict)
    # Example: {"liquidity_depth_drop_pct": 0.45, "affected_exchanges": ["binance", "okx"]}
    
    # Recommendations
    recommended_action: Optional[str] = None  # "reduce_exposure", "halt_trading", "monitor"
    
    # Related entities
    affected_symbols: List[str] = Field(default_factory=list)
    affected_exchanges: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-04T10:00:00Z",
                "level": "WARNING",
                "risk_type": "correlation_spike",
                "description": "Average correlation spiked from 0.35 to 0.78 in 1 hour",
                "severity_score": 0.65,
                "recommended_action": "reduce_exposure"
            }
        }


class GlobalRiskSignal(BaseModel):
    """Aggregated global risk signal from all risk engines
    
    This is the final output of RiskOrchestrator, combining:
    - Exposure matrix analysis
    - VaR/ES calculations
    - Systemic risk detection
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Overall risk level
    risk_level: RiskLevel  # Highest level from all components
    
    # Component results
    snapshot: RiskSnapshot
    exposure_matrix: ExposureMatrix
    var_result: Optional[VaRResult] = None
    es_result: Optional[ESResult] = None
    systemic_signals: List[SystemicRiskSignal] = Field(default_factory=list)
    
    # Aggregated metrics
    overall_risk_score: float = Field(ge=0.0, le=1.0)  # 0 = safe, 1 = extreme danger
    
    # ESS v3 integration
    ess_tier_recommendation: Optional[str] = None  # "NORMAL", "REDUCED", "EMERGENCY"
    ess_action_required: bool = False
    
    # Federation AI CRO integration
    cro_alert_sent: bool = False
    cro_approval_required: bool = False
    
    # Summary
    risk_summary: str  # Human-readable summary
    critical_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-04T10:00:00Z",
                "risk_level": "WARNING",
                "overall_risk_score": 0.65,
                "ess_tier_recommendation": "REDUCED",
                "risk_summary": "High correlation detected across BTC/ETH positions",
                "critical_issues": [],
                "warnings": ["Symbol concentration > 60%", "VaR exceeded 95% threshold"]
            }
        }


class RiskThreshold(BaseModel):
    """Risk threshold configuration"""
    name: str
    value: float
    breached: bool = False
    current_value: Optional[float] = None
    severity: RiskLevel = RiskLevel.INFO


class RiskLimits(BaseModel):
    """Complete set of risk limits from PolicyStore"""
    max_leverage: float = 5.0
    max_position_size_usd: float = 10000.0
    max_daily_drawdown_pct: float = 5.0
    max_correlation: float = 0.80
    var_95_limit: float = 1000.0
    var_99_limit: float = 2000.0
    es_975_limit: float = 2500.0
    max_symbol_concentration: float = 0.60  # 60% max in single symbol
    max_exchange_concentration: float = 0.80  # 80% max on single exchange
    min_liquidity_score: float = 0.50  # Minimum acceptable liquidity
    
    # Systemic risk thresholds
    correlation_spike_threshold: float = 0.20  # 20% increase triggers warning
    volatility_spike_threshold: float = 2.0  # 2x baseline volatility triggers warning


# Export all models
__all__ = [
    "RiskLevel",
    "SystemicRiskType",
    "PositionExposure",
    "RiskSnapshot",
    "CorrelationMatrix",
    "ExposureMatrix",
    "VaRResult",
    "ESResult",
    "SystemicRiskSignal",
    "GlobalRiskSignal",
    "RiskThreshold",
    "RiskLimits",
]

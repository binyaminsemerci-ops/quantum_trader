"""
Dashboard API Models
Sprint 4

Data structures for dashboard snapshot and real-time events.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timezone
from enum import Enum


# ========== ENUMS ==========

class MarketRegime(str, Enum):
    """Market regime classification"""
    HIGH_VOL_TRENDING = "HIGH_VOL_TRENDING"
    LOW_VOL_TRENDING = "LOW_VOL_TRENDING"
    HIGH_VOL_RANGING = "HIGH_VOL_RANGING"
    LOW_VOL_RANGING = "LOW_VOL_RANGING"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


class ESSState(str, Enum):
    """ESS state enum"""
    ARMED = "ARMED"
    TRIPPED = "TRIPPED"
    COOLING = "COOLING"
    UNKNOWN = "UNKNOWN"


class ServiceStatus(str, Enum):
    """Service health status"""
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class SignalDirection(str, Enum):
    """Signal direction"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PositionSide(str, Enum):
    """Position side"""
    LONG = "LONG"
    SHORT = "SHORT"


# ========== POSITION DATA ==========

@dataclass
class DashboardPosition:
    """Single position for dashboard"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    value: float  # size * current_price
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "value": round(self.value, 2)
        }


# ========== SIGNAL DATA ==========

@dataclass
class DashboardSignal:
    """AI signal for dashboard"""
    timestamp: str
    symbol: str
    direction: SignalDirection
    confidence: float
    strategy: str  # "ensemble", "meta_lstm", "rl_sizing"
    target_size: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "strategy": self.strategy,
            "target_size": self.target_size
        }


# ========== PORTFOLIO DATA ==========

@dataclass
class DashboardPortfolio:
    """Portfolio snapshot for dashboard"""
    equity: float
    cash: float
    margin_used: float
    margin_available: float
    total_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    monthly_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    position_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "margin_used": round(self.margin_used, 2),
            "margin_available": round(self.margin_available, 2),
            "total_pnl": round(self.total_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "monthly_pnl": round(self.monthly_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "position_count": self.position_count
        }


# ========== STRATEGY DATA ==========

@dataclass
class DashboardRLSizing:
    """RL position sizing details"""
    symbol: str
    proposed_risk_pct: float
    capped_risk_pct: float
    proposed_leverage: float
    capped_leverage: float
    volatility_bucket: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "proposed_risk_pct": round(self.proposed_risk_pct, 2),
            "capped_risk_pct": round(self.capped_risk_pct, 2),
            "proposed_leverage": round(self.proposed_leverage, 1),
            "capped_leverage": round(self.capped_leverage, 1),
            "volatility_bucket": self.volatility_bucket
        }


@dataclass
class DashboardStrategy:
    """Strategy and AI decision insights"""
    active_strategy: str  # "TREND_FOLLOW_V3", "MEAN_REVERT", etc.
    regime: MarketRegime
    ensemble_scores: Dict[str, float]  # {"xgb": 0.73, "lgbm": 0.69, ...}
    rl_sizing: Optional[DashboardRLSizing] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_strategy": self.active_strategy,
            "regime": self.regime.value,
            "ensemble_scores": {k: round(v, 3) for k, v in self.ensemble_scores.items()},
            "rl_sizing": self.rl_sizing.to_dict() if self.rl_sizing else None
        }


# ========== RISK DATA ==========

@dataclass
class DashboardRisk:
    """Risk metrics for dashboard (extended with max thresholds)"""
    ess_state: ESSState
    ess_reason: Optional[str]
    ess_tripped_at: Optional[str]
    daily_pnl_pct: float  # Current daily PnL as %
    daily_drawdown_pct: float  # Current daily DD
    weekly_drawdown_pct: float
    max_drawdown_pct: float  # Historical max DD
    max_allowed_dd_pct: float  # Policy limit (e.g., -10.0)
    exposure_total: float
    exposure_long: float
    exposure_short: float
    exposure_net: float
    open_risk_pct: float  # Total risk from open positions
    max_risk_per_trade_pct: float  # Policy limit (e.g., 1.0)
    risk_limit_used_pct: float  # How much of daily loss limit used
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ess_state": self.ess_state.value,
            "ess_reason": self.ess_reason,
            "ess_tripped_at": self.ess_tripped_at,
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "daily_drawdown_pct": round(self.daily_drawdown_pct, 2),
            "weekly_drawdown_pct": round(self.weekly_drawdown_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_allowed_dd_pct": round(self.max_allowed_dd_pct, 2),
            "exposure_total": round(self.exposure_total, 2),
            "exposure_long": round(self.exposure_long, 2),
            "exposure_short": round(self.exposure_short, 2),
            "exposure_net": round(self.exposure_net, 2),
            "open_risk_pct": round(self.open_risk_pct, 2),
            "max_risk_per_trade_pct": round(self.max_risk_per_trade_pct, 2),
            "risk_limit_used_pct": round(self.risk_limit_used_pct, 2)
        }


# ========== SYSTEM HEALTH DATA ==========

@dataclass
class ServiceHealthInfo:
    """Single service health"""
    name: str
    status: ServiceStatus
    latency_ms: Optional[float] = None
    last_check: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(latency_ms, 1) if (latency_ms := self.latency_ms) else None,
            "last_check": self.last_check
        }


@dataclass
class DashboardSystemHealth:
    """System health for dashboard"""
    overall_status: ServiceStatus
    services: List[ServiceHealthInfo]
    alerts_count: int
    last_alert: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "services": [s.to_dict() for s in self.services],
            "alerts_count": self.alerts_count,
            "last_alert": self.last_alert
        }


# ========== DASHBOARD SNAPSHOT ==========

@dataclass
class DashboardSnapshot:
    """
    Complete dashboard snapshot.
    
    Aggregates data from all microservices for initial dashboard load.
    
    Schema version: 1 (Sprint 4 Del 3)
    """
    timestamp: str
    portfolio: DashboardPortfolio
    positions: List[DashboardPosition]
    signals: List[DashboardSignal]  # Last 10-20 signals
    risk: DashboardRisk
    system: DashboardSystemHealth
    strategy: Optional[DashboardStrategy] = None  # Sprint 4 Del 2
    schema_version: int = 1  # Sprint 4 Del 3: API versioning
    partial_data: bool = False  # Sprint 4 Del 3: Indicates if some services failed
    errors: List[str] = field(default_factory=list)  # Sprint 4 Del 3: Service errors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "partial_data": self.partial_data,
            "errors": self.errors,
            "portfolio": self.portfolio.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "signals": [s.to_dict() for s in self.signals],
            "risk": self.risk.to_dict(),
            "system": self.system.to_dict(),
            "strategy": self.strategy.to_dict() if self.strategy else None
        }


# ========== REAL-TIME EVENTS ==========

EventType = Literal[
    "position_updated",
    "pnl_updated",
    "signal_generated",
    "ess_state_changed",
    "health_alert",
    "trade_executed",
    "order_placed",
    "strategy_updated",  # Sprint 4 Del 2
    "rl_sizing_updated",  # Sprint 4 Del 2
    "regime_changed"  # Sprint 4 Del 2
]


@dataclass
class DashboardEvent:
    """
    Real-time event for WebSocket.
    
    Types:
    - position_updated: Position changed (new, closed, updated)
    - pnl_updated: PnL recalculation
    - signal_generated: New AI signal
    - ess_state_changed: ESS state transition
    - health_alert: Service health issue
    - trade_executed: Order filled
    - order_placed: New order placed
    """
    type: EventType
    timestamp: str
    payload: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload
        }


# ========== HELPER FUNCTIONS ==========

def create_position_updated_event(position: DashboardPosition) -> DashboardEvent:
    """Create position updated event."""
    return DashboardEvent(
        type="position_updated",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload=position.to_dict()
    )


def create_pnl_updated_event(portfolio: DashboardPortfolio) -> DashboardEvent:
    """Create pnl_updated event"""
    return DashboardEvent(
        type="pnl_updated",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload={
            "equity": portfolio.equity,
            "total_pnl": portfolio.total_pnl,
            "daily_pnl": portfolio.daily_pnl,
            "daily_pnl_pct": portfolio.daily_pnl_pct,
            "unrealized_pnl": portfolio.unrealized_pnl
        }
    )


def create_signal_generated_event(signal: DashboardSignal) -> DashboardEvent:
    """Create signal generated event."""
    return DashboardEvent(
        type="signal_generated",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload=signal.to_dict()
    )


def create_ess_state_changed_event(
    new_state: ESSState,
    reason: Optional[str] = None
) -> DashboardEvent:
    """Create ess_state_changed event"""
    return DashboardEvent(
        type="ess_state_changed",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload={
            "ess_state": new_state.value,
            "reason": reason
        }
    )


def create_health_alert_event(
    service: str,
    status: ServiceStatus,
    message: str
) -> DashboardEvent:
    """Create health_alert event"""
    return DashboardEvent(
        type="health_alert",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload={
            "service": service,
            "status": status.value,
            "message": message
        }
    )


def create_strategy_updated_event(
    strategy: DashboardStrategy
) -> DashboardEvent:
    """Create strategy_updated event"""
    return DashboardEvent(
        type="strategy_updated",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload=strategy.to_dict()
    )


def create_rl_sizing_updated_event(
    rl_sizing: DashboardRLSizing
) -> DashboardEvent:
    """Create rl_sizing_updated event"""
    return DashboardEvent(
        type="rl_sizing_updated",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload=rl_sizing.to_dict()
    )


def create_regime_changed_event(
    old_regime: MarketRegime,
    new_regime: MarketRegime
) -> DashboardEvent:
    """Create regime_changed event"""
    return DashboardEvent(
        type="regime_changed",
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload={
            "old_regime": old_regime.value,
            "new_regime": new_regime.value
        }
    )

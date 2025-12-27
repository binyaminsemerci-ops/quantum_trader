"""
Event types for Quantum Trader EventBus.

All events inherit from the base Event dataclass and include typed payloads
for better developer experience and type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum


class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class RiskMode(str, Enum):
    """Risk modes for trading."""
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


class StrategyLifecycle(str, Enum):
    """Strategy lifecycle stages."""
    BACKTEST = "BACKTEST"
    SHADOW = "SHADOW"
    LIVE = "LIVE"
    RETIRED = "RETIRED"


@dataclass
class Event:
    """
    Base event class for the EventBus.
    
    All events must have a type (for routing), timestamp (for ordering/logging),
    and a payload dictionary containing event-specific data.
    """
    type: str
    timestamp: datetime
    payload: dict[str, Any]
    
    def __post_init__(self):
        """Ensure timestamp is set if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class PolicyUpdatedEvent(Event):
    """
    Fired by MSC AI when global trading policy is updated.
    
    Payload fields:
        - risk_mode: str (AGGRESSIVE/NORMAL/DEFENSIVE)
        - allowed_strategies: list[str] (strategy IDs)
        - global_min_confidence: float
        - max_risk_per_trade: float
        - max_positions: int
        - updated_by: str (component name)
    """
    
    def __post_init__(self):
        """Set type after initialization."""
        super().__post_init__()
        if self.type is None or self.type == "":
            object.__setattr__(self, 'type', 'policy.updated')
    
    @classmethod
    def create(
        cls,
        risk_mode: RiskMode,
        allowed_strategies: list[str],
        global_min_confidence: float,
        max_risk_per_trade: float,
        max_positions: int,
        updated_by: str = "MSC_AI",
    ) -> "PolicyUpdatedEvent":
        """Create a PolicyUpdatedEvent with typed parameters."""
        return cls(
            type="policy.updated",
            timestamp=datetime.utcnow(),
            payload={
                "risk_mode": risk_mode.value,
                "allowed_strategies": allowed_strategies,
                "global_min_confidence": global_min_confidence,
                "max_risk_per_trade": max_risk_per_trade,
                "max_positions": max_positions,
                "updated_by": updated_by,
            }
        )


@dataclass
class StrategyPromotedEvent(Event):
    """
    Fired by SG AI when a strategy changes lifecycle stage.
    
    Payload fields:
        - strategy_id: str
        - from_stage: str (BACKTEST/SHADOW/LIVE/RETIRED)
        - to_stage: str
        - reason: str
        - metrics: dict (performance metrics that justified the promotion)
    """
    
    def __post_init__(self):
        """Set type after initialization."""
        super().__post_init__()
        if self.type is None or self.type == "":
            object.__setattr__(self, 'type', 'strategy.promoted')
    
    @classmethod
    def create(
        cls,
        strategy_id: str,
        from_stage: StrategyLifecycle,
        to_stage: StrategyLifecycle,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> "StrategyPromotedEvent":
        """Create a StrategyPromotedEvent with typed parameters."""
        return cls(
            type="strategy.promoted",
            timestamp=datetime.utcnow(),
            payload={
                "strategy_id": strategy_id,
                "from_stage": from_stage.value,
                "to_stage": to_stage.value,
                "reason": reason,
                "metrics": metrics or {},
            }
        )


@dataclass
class ModelPromotedEvent(Event):
    """
    Fired by CLM when a model version is promoted to production.
    
    Payload fields:
        - model_name: str (XGBoost/LightGBM/NHiTS/PatchTST)
        - old_version: str
        - new_version: str
        - metrics: dict (validation metrics)
        - shadow_performance: dict (how it performed in shadow mode)
    """
    
    def __post_init__(self):
        """Set type after initialization."""
        super().__post_init__()
        if self.type is None or self.type == "":
            object.__setattr__(self, 'type', 'model.promoted')
    
    @classmethod
    def create(
        cls,
        model_name: str,
        old_version: str,
        new_version: str,
        metrics: dict[str, float],
        shadow_performance: dict[str, float] | None = None,
    ) -> "ModelPromotedEvent":
        """Create a ModelPromotedEvent with typed parameters."""
        return cls(
            type="model.promoted",
            timestamp=datetime.utcnow(),
            payload={
                "model_name": model_name,
                "old_version": old_version,
                "new_version": new_version,
                "metrics": metrics,
                "shadow_performance": shadow_performance or {},
            }
        )


@dataclass
class HealthStatusChangedEvent(Event):
    """
    Fired by System Health Monitor when system health changes.
    
    Payload fields:
        - old_status: str (HEALTHY/DEGRADED/CRITICAL)
        - new_status: str
        - component: str (which component triggered the change)
        - reason: str
        - metrics: dict (relevant health metrics)
    """
    
    def __post_init__(self):
        """Set type after initialization."""
        super().__post_init__()
        if self.type is None or self.type == "":
            object.__setattr__(self, 'type', 'health.status_changed')
    
    @classmethod
    def create(
        cls,
        old_status: HealthStatus,
        new_status: HealthStatus,
        component: str,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> "HealthStatusChangedEvent":
        """Create a HealthStatusChangedEvent with typed parameters."""
        return cls(
            type="health.status_changed",
            timestamp=datetime.utcnow(),
            payload={
                "old_status": old_status.value,
                "new_status": new_status.value,
                "component": component,
                "reason": reason,
                "metrics": metrics or {},
            }
        )


@dataclass
class OpportunitiesUpdatedEvent(Event):
    """
    Fired by OppRank when new symbol rankings are available.
    
    Payload fields:
        - top_symbols: list[str] (ranked symbol list)
        - scores: dict[str, float] (symbol -> score mapping)
        - criteria: dict (what criteria were used for ranking)
        - excluded_count: int (how many symbols were filtered out)
    """
    
    def __post_init__(self):
        """Set type after initialization."""
        super().__post_init__()
        if self.type is None or self.type == "":
            object.__setattr__(self, 'type', 'opportunities.updated')
    
    @classmethod
    def create(
        cls,
        top_symbols: list[str],
        scores: dict[str, float],
        criteria: dict[str, Any] | None = None,
        excluded_count: int = 0,
    ) -> "OpportunitiesUpdatedEvent":
        """Create an OpportunitiesUpdatedEvent with typed parameters."""
        return cls(
            type="opportunities.updated",
            timestamp=datetime.utcnow(),
            payload={
                "top_symbols": top_symbols,
                "scores": scores,
                "criteria": criteria or {},
                "excluded_count": excluded_count,
            }
        )


@dataclass
class TradeExecutedEvent(Event):
    """
    Fired by Executor when a trade order is filled.
    
    Payload fields:
        - order_id: str
        - symbol: str
        - side: str (BUY/SELL)
        - size: float
        - price: float
        - strategy_id: str
        - model: str (which model generated the signal)
        - pnl: float (for closes)
    """
    
    def __post_init__(self):
        """Set type after initialization."""
        super().__post_init__()
        if self.type is None or self.type == "":
            object.__setattr__(self, 'type', 'trade.executed')
    
    @classmethod
    def create(
        cls,
        order_id: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        strategy_id: str,
        model: str | None = None,
        pnl: float | None = None,
    ) -> "TradeExecutedEvent":
        """Create a TradeExecutedEvent with typed parameters."""
        return cls(
            type="trade.executed",
            timestamp=datetime.utcnow(),
            payload={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "strategy_id": strategy_id,
                "model": model,
                "pnl": pnl,
            }
        )

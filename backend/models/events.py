"""Event data models for EventBus v2."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Base event model with common fields."""
    
    event_type: str
    trace_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = None  # Source service/module


class SignalEvent(BaseEvent):
    """AI signal generation event."""
    
    event_type: str = "ai.signal.generated"
    
    symbol: str
    action: str  # "LONG" or "SHORT"
    confidence: float
    entry_price: float
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeApprovalEvent(BaseEvent):
    """Risk management trade approval event."""
    
    event_type: str = "risk.signal.approved"
    
    symbol: str
    action: str
    approved: bool
    rejection_reason: Optional[str] = None
    adjusted_size: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderEvent(BaseEvent):
    """Order execution event."""
    
    event_type: str = "execution.order.submitted"
    
    symbol: str
    order_id: str
    action: str
    quantity: float
    price: float
    order_type: str  # "MARKET", "LIMIT", etc.
    status: str      # "PENDING", "FILLED", "FAILED"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PositionEvent(BaseEvent):
    """Position lifecycle event."""
    
    event_type: str = "portfolio.position.opened"
    
    symbol: str
    position_id: str
    action: str
    entry_price: float
    quantity: float
    leverage: float
    pnl_usd: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerformanceEvent(BaseEvent):
    """Portfolio performance metrics event."""
    
    event_type: str = "portfolio.performance"
    
    equity_usd: float
    daily_pnl: float
    daily_pnl_pct: float
    open_positions: int
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemEvent(BaseEvent):
    """System-level event (health, errors, etc.)."""
    
    event_type: str = "system.status"
    
    level: str  # "INFO", "WARNING", "ERROR", "CRITICAL"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)

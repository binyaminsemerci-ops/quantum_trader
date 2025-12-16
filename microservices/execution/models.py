"""
Execution Service - Pydantic Models

Request/response models for REST API and internal communication.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TradeStatus(str, Enum):
    """Trade status."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING_EXIT = "pending_exit"


# ============================================================================
# AI DECISION EVENT
# ============================================================================

class AIDecisionEvent(BaseModel):
    """Event published by ai-engine-service when trade decision is made."""
    symbol: str
    side: str  # buy/sell/long/short
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail_percent: Optional[float] = None
    position_size_usd: float
    leverage: Optional[int] = None
    model: str = "ensemble"
    meta_strategy: Optional[str] = None
    timestamp: str


# ============================================================================
# ORDER PLACEMENT
# ============================================================================

class OrderRequest(BaseModel):
    """Request to place an order (manual or from AI)."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: OrderSide = Field(..., description="Order side")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, description="Limit price (None for market)")
    leverage: Optional[int] = Field(None, ge=1, le=125, description="Leverage (1-125)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    trail_percent: Optional[float] = Field(None, description="Trailing stop percentage")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class OrderResponse(BaseModel):
    """Response after order placement."""
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    symbol: str
    side: str
    quantity: float
    price: Optional[float] = None
    status: OrderStatus
    message: Optional[str] = None
    timestamp: str


# ============================================================================
# POSITION & TRADE QUERIES
# ============================================================================

class Position(BaseModel):
    """Current position details."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int
    margin_usd: float
    liquidation_price: Optional[float] = None
    has_tp_sl: bool
    tp_orders: List[Dict[str, Any]] = []
    sl_orders: List[Dict[str, Any]] = []


class Trade(BaseModel):
    """Trade record from TradeStore."""
    trade_id: str
    symbol: str
    side: str
    status: TradeStatus
    quantity: float
    leverage: int
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl_usd: Optional[float] = None
    pnl_percent: Optional[float] = None
    model: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


class PositionListResponse(BaseModel):
    """Response for /api/execution/positions."""
    positions: List[Position]
    total_positions: int
    total_margin_usd: float
    total_unrealized_pnl: float


class TradeListResponse(BaseModel):
    """Response for /api/execution/trades."""
    trades: List[Trade]
    total_trades: int
    open_trades: int
    closed_trades: int


# ============================================================================
# HEALTH & METRICS
# ============================================================================

class ComponentHealth(BaseModel):
    """Health status of a component."""
    healthy: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ServiceHealth(BaseModel):
    """Overall service health."""
    service: str
    healthy: bool
    running: bool
    components: Dict[str, ComponentHealth]
    active_positions: int
    orders_last_minute: int
    timestamp: str


class ExecutionMetrics(BaseModel):
    """Execution performance metrics."""
    orders_placed_total: int
    orders_filled_total: int
    orders_failed_total: int
    success_rate_pct: float
    avg_order_latency_ms: float
    avg_slippage_pct: float
    rate_limit_tokens_available: int
    rate_limit_tokens_per_minute: int

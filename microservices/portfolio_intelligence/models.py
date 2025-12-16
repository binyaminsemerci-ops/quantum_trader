"""
Portfolio Intelligence Service - Data Models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class PositionSide(str, Enum):
    """Position side."""
    LONG = "LONG"
    SHORT = "SHORT"


class SymbolCategory(str, Enum):
    """Symbol quality category."""
    CORE = "CORE"
    EXPANSION = "EXPANSION"
    MONITORING = "MONITORING"
    TOXIC = "TOXIC"


# ============================================================================
# EVENT MODELS (IN)
# ============================================================================

class TradeOpenedEvent(BaseModel):
    """Event: trade.opened"""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: str


class TradeClosedEvent(BaseModel):
    """Event: trade.closed"""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    realized_pnl_pct: float
    duration_sec: float
    timestamp: str


class OrderExecutedEvent(BaseModel):
    """Event: order.executed"""
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    timestamp: str


class ESSTrippedEvent(BaseModel):
    """Event: ess.tripped"""
    reason: str
    severity: str
    timestamp: str


class MarketTickEvent(BaseModel):
    """Event: market.tick"""
    symbol: str
    price: float
    timestamp: str


# ============================================================================
# EVENT MODELS (OUT)
# ============================================================================

class PortfolioSnapshotUpdatedEvent(BaseModel):
    """Event: portfolio.snapshot_updated"""
    total_equity: float
    cash_balance: float
    total_exposure: float
    num_positions: int
    unrealized_pnl: float
    realized_pnl_today: float
    daily_drawdown_pct: float
    timestamp: str


class PortfolioPnLUpdatedEvent(BaseModel):
    """Event: portfolio.pnl_updated"""
    realized_pnl_today: float
    realized_pnl_total: float
    unrealized_pnl: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    timestamp: str


class PortfolioDrawdownUpdatedEvent(BaseModel):
    """Event: portfolio.drawdown_updated"""
    daily_drawdown_pct: float
    weekly_drawdown_pct: float
    max_drawdown_pct: float
    peak_equity: float
    current_equity: float
    timestamp: str


class PortfolioExposureUpdatedEvent(BaseModel):
    """Event: portfolio.exposure_updated"""
    total_exposure: float
    exposure_by_symbol: Dict[str, float]
    exposure_by_sector: Dict[str, float]
    long_exposure: float
    short_exposure: float
    net_exposure: float
    timestamp: str


# ============================================================================
# API RESPONSE MODELS
# ============================================================================

class PositionInfo(BaseModel):
    """Open position information."""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    exposure: float
    leverage: float
    category: str = "EXPANSION"


class PortfolioSnapshot(BaseModel):
    """Complete portfolio snapshot."""
    total_equity: float
    cash_balance: float
    total_exposure: float
    num_positions: int
    positions: List[PositionInfo]
    unrealized_pnl: float
    realized_pnl_today: float
    daily_pnl: float
    daily_drawdown_pct: float
    timestamp: str


class PnLBreakdown(BaseModel):
    """PnL breakdown."""
    realized_pnl_total: float
    realized_pnl_today: float
    realized_pnl_week: float
    realized_pnl_month: float
    unrealized_pnl: float
    total_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    win_rate: float
    profit_factor: float
    timestamp: str


class ExposureBreakdown(BaseModel):
    """Exposure breakdown by symbol/sector."""
    total_exposure: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    exposure_by_symbol: Dict[str, float]
    exposure_by_sector: Dict[str, float]
    exposure_pct_of_equity: float
    timestamp: str


class DrawdownMetrics(BaseModel):
    """Drawdown metrics."""
    daily_drawdown_pct: float
    weekly_drawdown_pct: float
    max_drawdown_pct: float
    peak_equity: float
    current_equity: float
    days_since_peak: int
    recovery_progress_pct: float
    timestamp: str


class ComponentHealth(BaseModel):
    """Component health status."""
    name: str
    status: str
    message: Optional[str] = None


class ServiceHealth(BaseModel):
    """Service health response."""
    service: str
    version: str
    status: str
    uptime_seconds: float
    components: List[ComponentHealth]
    snapshot_updates: int
    last_snapshot_update: Optional[str] = None
    timestamp: str

"""
Exchange-Agnostic Models

EPIC-EXCH-001: Pydantic models for multi-exchange trading.
All models are exchange-neutral and can be mapped to/from
Binance, Bybit, OKX, and other exchange formats.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    """Order side (direction)"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class TimeInForce(str, Enum):
    """Time in force for limit orders"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Good Till Crossing (Post-Only)


class OrderStatus(str, Enum):
    """Order status"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    """Position side for futures"""
    BOTH = "BOTH"  # One-way mode
    LONG = "LONG"  # Hedge mode
    SHORT = "SHORT"  # Hedge mode


# ============================================================================
# REQUEST MODELS
# ============================================================================

class OrderRequest(BaseModel):
    """
    Unified order request model.
    
    Can be mapped to any exchange's order format.
    """
    symbol: str = Field(..., description="Trading pair (e.g., BTCUSDT)")
    side: OrderSide = Field(..., description="BUY or SELL")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    
    # Optional fields
    price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop/trigger price")
    time_in_force: Optional[TimeInForce] = Field(TimeInForce.GTC, description="Time in force")
    reduce_only: bool = Field(False, description="Reduce-only order (futures)")
    position_side: Optional[PositionSide] = Field(None, description="Position side (futures hedge mode)")
    
    # Advanced options
    client_order_id: Optional[str] = Field(None, description="Custom order ID")
    leverage: Optional[int] = Field(None, ge=1, le=125, description="Leverage (futures)")
    
    # Risk management
    take_profit_price: Optional[Decimal] = Field(None, gt=0, description="TP price")
    stop_loss_price: Optional[Decimal] = Field(None, gt=0, description="SL price")
    trailing_stop_pct: Optional[Decimal] = Field(None, gt=0, le=100, description="Trailing stop %")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase"""
        return v.upper()
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": "0.001",
                "reduce_only": False,
            }
        }


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class OrderResult(BaseModel):
    """
    Unified order result model.
    
    Returned after successful order placement.
    """
    order_id: str = Field(..., description="Exchange order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    symbol: str = Field(..., description="Trading pair")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., description="Order quantity")
    filled_quantity: Decimal = Field(Decimal("0"), description="Filled quantity")
    price: Optional[Decimal] = Field(None, description="Order price")
    average_price: Optional[Decimal] = Field(None, description="Average fill price")
    status: OrderStatus = Field(..., description="Order status")
    timestamp: datetime = Field(..., description="Order timestamp")
    
    # Exchange-specific metadata
    exchange: str = Field(..., description="Exchange name (binance, bybit, okx)")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Original exchange response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "123456789",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": "0.001",
                "filled_quantity": "0.001",
                "average_price": "50000.00",
                "status": "FILLED",
                "timestamp": "2024-12-04T12:00:00Z",
                "exchange": "binance",
            }
        }


class CancelResult(BaseModel):
    """
    Unified order cancellation result.
    """
    order_id: str = Field(..., description="Canceled order ID")
    symbol: str = Field(..., description="Trading pair")
    status: OrderStatus = Field(..., description="Order status after cancel")
    success: bool = Field(..., description="Whether cancel succeeded")
    message: Optional[str] = Field(None, description="Cancel result message")
    exchange: str = Field(..., description="Exchange name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "123456789",
                "symbol": "BTCUSDT",
                "status": "CANCELED",
                "success": True,
                "exchange": "binance",
            }
        }


class Position(BaseModel):
    """
    Unified position model (for futures).
    """
    symbol: str = Field(..., description="Trading pair")
    side: OrderSide = Field(..., description="Position side (LONG=BUY, SHORT=SELL)")
    quantity: Decimal = Field(..., description="Position size (abs value)")
    entry_price: Decimal = Field(..., gt=0, description="Average entry price")
    mark_price: Decimal = Field(..., gt=0, description="Current mark price")
    liquidation_price: Optional[Decimal] = Field(None, description="Liquidation price")
    
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    realized_pnl: Decimal = Field(Decimal("0"), description="Realized P&L")
    
    leverage: int = Field(..., ge=1, description="Position leverage")
    margin: Decimal = Field(..., gt=0, description="Position margin")
    
    # Risk levels
    take_profit: Optional[Decimal] = Field(None, description="TP price")
    stop_loss: Optional[Decimal] = Field(None, description="SL price")
    
    # Metadata
    exchange: str = Field(..., description="Exchange name")
    position_side: Optional[PositionSide] = Field(PositionSide.BOTH, description="Position side mode")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Position timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": "0.1",
                "entry_price": "50000.00",
                "mark_price": "51000.00",
                "unrealized_pnl": "100.00",
                "leverage": 10,
                "margin": "500.00",
                "exchange": "binance",
            }
        }


class Balance(BaseModel):
    """
    Unified balance model.
    """
    asset: str = Field(..., description="Asset name (e.g., USDT, BTC)")
    free: Decimal = Field(..., ge=0, description="Available balance")
    locked: Decimal = Field(Decimal("0"), ge=0, description="Locked balance")
    total: Decimal = Field(..., ge=0, description="Total balance (free + locked)")
    
    # Optional fields
    unrealized_pnl: Optional[Decimal] = Field(None, description="Unrealized P&L")
    margin_balance: Optional[Decimal] = Field(None, description="Margin balance (futures)")
    
    exchange: str = Field(..., description="Exchange name")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Balance timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "asset": "USDT",
                "free": "10000.00",
                "locked": "500.00",
                "total": "10500.00",
                "exchange": "binance",
            }
        }


class Kline(BaseModel):
    """
    Unified candlestick/kline model.
    """
    symbol: str = Field(..., description="Trading pair")
    interval: str = Field(..., description="Timeframe (1m, 5m, 1h, 1d, etc.)")
    open_time: datetime = Field(..., description="Candle open time")
    close_time: datetime = Field(..., description="Candle close time")
    
    open: Decimal = Field(..., gt=0, description="Open price")
    high: Decimal = Field(..., gt=0, description="High price")
    low: Decimal = Field(..., gt=0, description="Low price")
    close: Decimal = Field(..., gt=0, description="Close price")
    
    volume: Decimal = Field(..., ge=0, description="Volume")
    quote_volume: Decimal = Field(..., ge=0, description="Quote asset volume")
    trades: int = Field(..., ge=0, description="Number of trades")
    
    exchange: str = Field(..., description="Exchange name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "open_time": "2024-12-04T12:00:00Z",
                "close_time": "2024-12-04T13:00:00Z",
                "open": "50000.00",
                "high": "51000.00",
                "low": "49500.00",
                "close": "50500.00",
                "volume": "100.5",
                "quote_volume": "5050000.00",
                "trades": 1234,
                "exchange": "binance",
            }
        }

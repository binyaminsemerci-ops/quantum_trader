"""
Order Domain Models
EPIC: DASHBOARD-V3-TRADING-PANELS

Defines order record structure for dashboard display.
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class OrderRecord(BaseModel):
    """
    Order record for dashboard display.
    
    Maps to TradeLog database model.
    """
    id: int = Field(description="Order ID")
    timestamp: datetime = Field(description="Order timestamp")
    account: str = Field(default="default", description="Trading account")
    exchange: str = Field(default="binance_testnet", description="Exchange")
    symbol: str = Field(description="Trading symbol")
    side: str = Field(description="BUY or SELL")
    order_type: str = Field(default="MARKET", description="Order type")
    size: float = Field(description="Order quantity")
    price: float = Field(description="Order price")
    status: OrderStatus = Field(description="Order status")
    strategy_id: Optional[str] = Field(default=None, description="Strategy identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

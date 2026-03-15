from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    message: str

# AI Hedge Fund Dashboard Schemas
class AIStatus(BaseModel):
    accuracy: float
    sharpe: float
    latency: int
    models: List[str]

class Prediction(BaseModel):
    id: str
    timestamp: str
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    model: str
    reason: str
    volatility: float
    regime: str
    position_size_usd: float

class PredictionsResponse(BaseModel):
    predictions: List[Prediction]
    count: int
    timestamp: float

class Portfolio(BaseModel):
    pnl: float
    exposure: float
    drawdown: float
    positions: int

class Risk(BaseModel):
    var: float
    cvar: float
    volatility: float
    regime: str

class SystemHealth(BaseModel):
    cpu: float
    ram: float
    uptime: int
    containers: int

# Trade schemas (for future database integration)
class TradeBase(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float

class Trade(TradeBase):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True


# Journal schemas
class JournalEntryCreate(BaseModel):
    trade_symbol: str
    trade_side: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    strategy_tag: Optional[str] = None
    notes: Optional[str] = None
    rating: Optional[int] = None
    mistakes: Optional[str] = None
    lessons: Optional[str] = None

class JournalEntryResponse(JournalEntryCreate):
    id: int
    created_by: str
    created_at: datetime

    class Config:
        from_attributes = True


# Incident schemas
class IncidentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    severity: str = "medium"
    category: Optional[str] = None
    affected_services: Optional[str] = None

class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None

class IncidentResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    severity: str
    status: str
    category: Optional[str]
    affected_services: Optional[str]
    root_cause: Optional[str]
    resolution: Optional[str]
    reported_by: str
    assigned_to: Optional[str]
    opened_at: datetime
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]

    class Config:
        from_attributes = True

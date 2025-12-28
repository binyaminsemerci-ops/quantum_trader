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

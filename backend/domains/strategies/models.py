"""
Strategy Domain Models
EPIC: DASHBOARD-V3-TRADING-PANELS

Defines strategy information structure for dashboard display.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class StrategyInfo(BaseModel):
    """
    Strategy information for dashboard display.
    
    Represents an active trading strategy configuration.
    """
    name: str = Field(description="Strategy name/ID")
    enabled: bool = Field(description="Whether strategy is active")
    profile: str = Field(description="Risk profile (micro/low/normal/agg)")
    exchanges: List[str] = Field(default_factory=list, description="Target exchanges")
    symbols: List[str] = Field(default_factory=list, description="Target symbols")
    description: Optional[str] = Field(default=None, description="Strategy description")
    min_confidence: Optional[float] = Field(default=None, description="Minimum confidence threshold")

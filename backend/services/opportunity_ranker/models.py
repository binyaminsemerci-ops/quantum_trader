"""
Data models for Opportunity Ranker.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RankingCriteria:
    """Criteria used for ranking symbols."""
    min_volume: float = 1e9  # Minimum 24h volume
    min_liquidity_score: float = 0.5
    trend_weight: float = 0.35
    volatility_weight: float = 0.25
    liquidity_weight: float = 0.20
    performance_weight: float = 0.20
    
    def to_dict(self) -> dict:
        return {
            "min_volume": self.min_volume,
            "min_liquidity_score": self.min_liquidity_score,
            "trend_weight": self.trend_weight,
            "volatility_weight": self.volatility_weight,
            "liquidity_weight": self.liquidity_weight,
            "performance_weight": self.performance_weight,
        }


@dataclass
class SymbolScore:
    """
    Score breakdown for a trading symbol.
    
    Contains individual component scores and composite score.
    """
    symbol: str
    
    # Component scores (0-1)
    trend_score: float
    volatility_score: float
    liquidity_score: float
    performance_score: float
    
    # Composite score (0-1)
    total_score: float
    
    # Supporting data
    volume_24h: float
    atr: float
    trend_strength: float
    
    timestamp: datetime
    
    @classmethod
    def calculate(
        cls,
        symbol: str,
        trend_score: float,
        volatility_score: float,
        liquidity_score: float,
        performance_score: float,
        criteria: RankingCriteria,
        volume_24h: float = 0,
        atr: float = 0,
        trend_strength: float = 0,
    ) -> "SymbolScore":
        """Calculate total score using criteria weights."""
        total = (
            trend_score * criteria.trend_weight +
            volatility_score * criteria.volatility_weight +
            liquidity_score * criteria.liquidity_weight +
            performance_score * criteria.performance_weight
        )
        
        return cls(
            symbol=symbol,
            trend_score=trend_score,
            volatility_score=volatility_score,
            liquidity_score=liquidity_score,
            performance_score=performance_score,
            total_score=total,
            volume_24h=volume_24h,
            atr=atr,
            trend_strength=trend_strength,
            timestamp=datetime.now(),
        )
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "trend_score": self.trend_score,
            "volatility_score": self.volatility_score,
            "liquidity_score": self.liquidity_score,
            "performance_score": self.performance_score,
            "total_score": self.total_score,
            "volume_24h": self.volume_24h,
            "atr": self.atr,
            "trend_strength": self.trend_strength,
            "timestamp": self.timestamp.isoformat(),
        }

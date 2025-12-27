"""
Data models for Meta Strategy Controller.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class MarketAnalysis:
    """
    Market condition analysis result.
    
    Used by MSC AI to determine optimal trading policy.
    """
    regime: MarketRegime
    volatility: float  # ATR-based
    trend_strength: float  # 0-1
    correlation: float  # Market correlation
    liquidity_score: float  # 0-1
    risk_score: float  # 0-1 (1 = high risk)
    
    # Recent performance metrics
    recent_win_rate: float
    recent_sharpe: float
    recent_drawdown: float
    
    timestamp: datetime
    
    def is_favorable_for_aggressive(self) -> bool:
        """Determine if conditions favor aggressive trading."""
        return (
            self.trend_strength > 0.7
            and self.volatility > 0.5
            and self.risk_score < 0.4
            and self.recent_sharpe > 1.5
            and self.recent_drawdown > -3.0
        )
    
    def is_unfavorable_requires_defensive(self) -> bool:
        """Determine if conditions require defensive mode."""
        return (
            self.risk_score > 0.7
            or self.recent_drawdown < -5.0
            or (self.regime == MarketRegime.CHOPPY and self.volatility > 0.8)
            or self.recent_win_rate < 0.4
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "correlation": self.correlation,
            "liquidity_score": self.liquidity_score,
            "risk_score": self.risk_score,
            "recent_win_rate": self.recent_win_rate,
            "recent_sharpe": self.recent_sharpe,
            "recent_drawdown": self.recent_drawdown,
            "timestamp": self.timestamp.isoformat(),
        }

"""
POSITION INTELLIGENCE LAYER (PIL)
==================================

Classifies positions and provides intelligence on position lifecycle.

Categories:
- POTENTIAL_WINNER: Early gains, strong momentum
- WINNER: Sustained profits, exceeding targets
- STRUGGLING: Small losses, consolidating
- LOSER: Significant losses, needs attention
- BREAKEVEN: Near entry, neutral

Recommendations:
- HOLD: Continue monitoring
- SCALE_IN: Add to position (if winning)
- REDUCE: Partial exit
- EXIT: Full exit recommended
- TIGHTEN_SL: Move stop closer
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PositionCategory(Enum):
    """Position classification categories"""
    POTENTIAL_WINNER = "potential_winner"
    WINNER = "winner"
    STRUGGLING = "struggling"
    LOSER = "loser"
    BREAKEVEN = "breakeven"
    UNKNOWN = "unknown"


class PositionRecommendation(Enum):
    """Actions recommended for position"""
    HOLD = "hold"
    SCALE_IN = "scale_in"
    REDUCE = "reduce"
    EXIT = "exit"
    TIGHTEN_SL = "tighten_sl"
    EXTEND_HOLD = "extend_hold"


@dataclass
class PositionClassification:
    """Classification result for a position"""
    symbol: str
    category: PositionCategory
    recommendation: PositionRecommendation
    confidence: float  # 0.0 to 1.0
    rationale: str
    metrics: Dict[str, Any]
    timestamp: datetime


class PositionIntelligenceLayer:
    """
    Analyzes open positions and provides intelligence on their state.
    
    Integration:
    - Called by position_monitor.py after each position update
    - Provides classification to PAL for amplification decisions
    - Informs SafetyGovernor on position health
    """
    
    def __init__(self):
        """Initialize Position Intelligence Layer"""
        self.logger = logging.getLogger(__name__)
        self.classifications: Dict[str, PositionClassification] = {}
        
        # Thresholds
        self.winner_threshold = 0.03  # 3% profit
        self.potential_winner_threshold = 0.01  # 1% profit
        self.struggling_threshold = -0.02  # -2% loss
        self.loser_threshold = -0.05  # -5% loss
        
        # Time thresholds (minutes)
        self.early_exit_time = 30
        self.extended_hold_time = 120
        
        self.logger.info("📊 [PIL] Position Intelligence Layer initialized")
    
    def classify_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_age_minutes: int,
        unrealized_pnl_pct: float,
        volume_profile: Optional[Dict[str, Any]] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> PositionClassification:
        """
        Classify a position based on performance and context.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            current_price: Current market price
            position_age_minutes: Age of position in minutes
            unrealized_pnl_pct: Unrealized PnL percentage
            volume_profile: Optional volume analysis
            market_conditions: Optional market state
        
        Returns:
            PositionClassification with category and recommendation
        """
        # Determine category
        if unrealized_pnl_pct >= self.winner_threshold:
            category = PositionCategory.WINNER
        elif unrealized_pnl_pct >= self.potential_winner_threshold:
            category = PositionCategory.POTENTIAL_WINNER
        elif unrealized_pnl_pct >= self.struggling_threshold:
            if abs(unrealized_pnl_pct) < 0.005:  # Within 0.5%
                category = PositionCategory.BREAKEVEN
            else:
                category = PositionCategory.STRUGGLING
        elif unrealized_pnl_pct >= self.loser_threshold:
            category = PositionCategory.STRUGGLING
        else:
            category = PositionCategory.LOSER
        
        # Determine recommendation
        recommendation, confidence, rationale = self._generate_recommendation(
            category=category,
            unrealized_pnl_pct=unrealized_pnl_pct,
            position_age_minutes=position_age_minutes,
            volume_profile=volume_profile,
            market_conditions=market_conditions
        )
        
        # Create classification
        classification = PositionClassification(
            symbol=symbol,
            category=category,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            metrics={
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "position_age_minutes": position_age_minutes,
                "entry_price": entry_price,
                "current_price": current_price
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store classification
        self.classifications[symbol] = classification
        
        self.logger.info(
            f"📊 [PIL] {symbol}: {category.value.upper()} → {recommendation.value.upper()} "
            f"(PnL: {unrealized_pnl_pct:+.2%}, Age: {position_age_minutes}m, Conf: {confidence:.0%})"
        )
        
        return classification
    
    def _generate_recommendation(
        self,
        category: PositionCategory,
        unrealized_pnl_pct: float,
        position_age_minutes: int,
        volume_profile: Optional[Dict[str, Any]],
        market_conditions: Optional[Dict[str, Any]]
    ) -> tuple[PositionRecommendation, float, str]:
        """
        Generate recommendation based on classification and context.
        
        Returns:
            (recommendation, confidence, rationale)
        """
        if category == PositionCategory.WINNER:
            # Strong winner - consider scaling or holding
            if position_age_minutes < self.early_exit_time:
                return (
                    PositionRecommendation.HOLD,
                    0.9,
                    f"Strong winner ({unrealized_pnl_pct:+.1%}) still early, let it run"
                )
            elif unrealized_pnl_pct > 0.10:  # 10%+ profit
                return (
                    PositionRecommendation.REDUCE,
                    0.8,
                    f"Massive gain ({unrealized_pnl_pct:+.1%}) - consider partial profit"
                )
            else:
                return (
                    PositionRecommendation.EXTEND_HOLD,
                    0.85,
                    f"Winner ({unrealized_pnl_pct:+.1%}) with good momentum"
                )
        
        elif category == PositionCategory.POTENTIAL_WINNER:
            # Early winner - hold and monitor
            return (
                PositionRecommendation.HOLD,
                0.75,
                f"Early profit ({unrealized_pnl_pct:+.1%}) - monitor for expansion"
            )
        
        elif category == PositionCategory.BREAKEVEN:
            # Near entry - patience
            if position_age_minutes > self.extended_hold_time:
                return (
                    PositionRecommendation.TIGHTEN_SL,
                    0.7,
                    f"Stuck at breakeven for {position_age_minutes}m - tighten protection"
                )
            else:
                return (
                    PositionRecommendation.HOLD,
                    0.6,
                    f"At breakeven ({unrealized_pnl_pct:+.1%}) - needs more time"
                )
        
        elif category == PositionCategory.STRUGGLING:
            # Small loss - give it time or tighten SL
            if position_age_minutes < 15:
                return (
                    PositionRecommendation.HOLD,
                    0.65,
                    f"Small loss ({unrealized_pnl_pct:+.1%}) but very early"
                )
            elif unrealized_pnl_pct < -0.03:  # -3%+
                return (
                    PositionRecommendation.TIGHTEN_SL,
                    0.8,
                    f"Struggling ({unrealized_pnl_pct:+.1%}) - protect capital"
                )
            else:
                return (
                    PositionRecommendation.HOLD,
                    0.6,
                    f"Minor loss ({unrealized_pnl_pct:+.1%}) - monitor closely"
                )
        
        elif category == PositionCategory.LOSER:
            # Significant loss - consider exit
            if unrealized_pnl_pct < -0.08:  # -8%+
                return (
                    PositionRecommendation.EXIT,
                    0.9,
                    f"Heavy loss ({unrealized_pnl_pct:+.1%}) - cut losses"
                )
            else:
                return (
                    PositionRecommendation.REDUCE,
                    0.85,
                    f"Losing position ({unrealized_pnl_pct:+.1%}) - reduce exposure"
                )
        
        else:
            return (
                PositionRecommendation.HOLD,
                0.5,
                "Insufficient data for recommendation"
            )
    
    def get_classification(self, symbol: str) -> Optional[PositionClassification]:
        """Get latest classification for a symbol"""
        return self.classifications.get(symbol)
    
    def get_portfolio_health(self) -> Dict[str, Any]:
        """
        Analyze overall portfolio health based on position classifications.
        
        Returns:
            Portfolio health metrics
        """
        if not self.classifications:
            return {
                "status": "EMPTY",
                "total_positions": 0,
                "winners": 0,
                "losers": 0,
                "health_score": 0.0
            }
        
        # Count by category
        categories = {cat: 0 for cat in PositionCategory}
        for classification in self.classifications.values():
            categories[classification.category] += 1
        
        total = len(self.classifications)
        winners = categories[PositionCategory.WINNER] + categories[PositionCategory.POTENTIAL_WINNER]
        losers = categories[PositionCategory.LOSER] + categories[PositionCategory.STRUGGLING]
        
        # Health score (0.0 to 1.0)
        health_score = (winners - losers) / total if total > 0 else 0.5
        health_score = max(0.0, min(1.0, (health_score + 1) / 2))
        
        # Determine status
        if health_score >= 0.7:
            status = "EXCELLENT"
        elif health_score >= 0.5:
            status = "GOOD"
        elif health_score >= 0.3:
            status = "FAIR"
        else:
            status = "POOR"
        
        return {
            "status": status,
            "total_positions": total,
            "winners": winners,
            "losers": losers,
            "breakeven": categories[PositionCategory.BREAKEVEN],
            "health_score": health_score,
            "categories": {cat.value: count for cat, count in categories.items()}
        }
    
    def clear_classification(self, symbol: str):
        """Remove classification for closed position"""
        if symbol in self.classifications:
            del self.classifications[symbol]
            self.logger.debug(f"📊 [PIL] Cleared classification for {symbol}")


# Global singleton
_position_intelligence: Optional[PositionIntelligenceLayer] = None


def get_position_intelligence() -> PositionIntelligenceLayer:
    """Get or create Position Intelligence Layer singleton"""
    global _position_intelligence
    if _position_intelligence is None:
        _position_intelligence = PositionIntelligenceLayer()
    return _position_intelligence

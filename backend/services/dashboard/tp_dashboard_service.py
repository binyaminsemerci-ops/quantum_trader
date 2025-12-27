"""
TP Dashboard Service

Service facade for TP v3 analytics dashboard.
Aggregates data from TPPerformanceTracker, tp_profiles_v3, and TPOptimizerV3.

Provides:
- List of tracked (strategy_id, symbol) pairs
- Combined metrics + profile + recommendation per pair
- Best/worst performance summaries
"""

import logging
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

from backend.services.monitoring.tp_performance_tracker import (
    TPPerformanceTracker,
    TPMetrics,
    get_tp_tracker
)
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
    TPProfile,
    TrailingProfile,
    MarketRegime,
    get_tp_and_trailing_profile
)
from backend.services.monitoring.tp_optimizer_v3 import (
    TPOptimizerV3,
    TPAdjustmentRecommendation
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class TPDashboardKey(BaseModel):
    """Key identifying a strategy/symbol pair."""
    strategy_id: str = Field(..., description="Strategy identifier")
    symbol: str = Field(..., description="Trading symbol")
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy_id": "RL_V3",
                "symbol": "BTCUSDT"
            }
        }


class TPDashboardProfileLeg(BaseModel):
    """Single leg in TP profile."""
    label: str = Field(..., description="Leg label (e.g., 'TP1')")
    r_multiple: float = Field(..., description="Risk multiple")
    size_fraction: float = Field(..., description="Position fraction")
    kind: str = Field(..., description="Execution type: HARD or SOFT")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "TP1",
                "r_multiple": 1.0,
                "size_fraction": 0.30,
                "kind": "HARD"
            }
        }


class TPDashboardProfile(BaseModel):
    """TP profile configuration."""
    profile_id: str = Field(..., description="Profile identifier")
    legs: List[TPDashboardProfileLeg] = Field(..., description="TP legs")
    trailing_profile_id: Optional[str] = Field(
        None,
        description="Trailing profile ID if trailing is enabled"
    )
    description: str = Field("", description="Profile description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "TREND_DEFAULT",
                "legs": [
                    {"label": "TP1", "r_multiple": 0.5, "size_fraction": 0.15, "kind": "SOFT"},
                    {"label": "TP2", "r_multiple": 1.0, "size_fraction": 0.20, "kind": "HARD"}
                ],
                "trailing_profile_id": "TREND_TRAILING",
                "description": "Trend-following profile"
            }
        }


class TPDashboardMetrics(BaseModel):
    """TP performance metrics."""
    tp_hit_rate: float = Field(..., description="TP hit rate (0.0-1.0)")
    tp_attempts: int = Field(..., description="Total attempts")
    tp_hits: int = Field(..., description="Successful hits")
    tp_misses: int = Field(..., description="Misses")
    avg_r_multiple: Optional[float] = Field(None, description="Average R multiple")
    avg_slippage_pct: Optional[float] = Field(None, description="Average slippage %")
    max_slippage_pct: Optional[float] = Field(None, description="Max slippage %")
    avg_time_to_tp_minutes: Optional[float] = Field(None, description="Avg time to TP (min)")
    total_tp_profit_usd: Optional[float] = Field(None, description="Total profit (USD)")
    avg_tp_profit_usd: Optional[float] = Field(None, description="Avg profit per hit (USD)")
    premature_exits: Optional[int] = Field(None, description="Premature exit count")
    missed_opportunities_usd: Optional[float] = Field(None, description="Missed profit (USD)")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tp_hit_rate": 0.62,
                "tp_attempts": 45,
                "tp_hits": 28,
                "tp_misses": 17,
                "avg_r_multiple": 1.61,
                "avg_slippage_pct": 0.008,
                "total_tp_profit_usd": 1245.50
            }
        }


class TPDashboardRecommendation(BaseModel):
    """TP optimization recommendation."""
    has_recommendation: bool = Field(..., description="Whether recommendation exists")
    profile_id: Optional[str] = Field(None, description="Profile being evaluated")
    suggested_scale_factor: Optional[float] = Field(None, description="Suggested scale factor")
    reason: Optional[str] = Field(None, description="Recommendation reason")
    confidence: Optional[float] = Field(None, description="Confidence (0.0-1.0)")
    direction: Optional[str] = Field(None, description="CLOSER, FURTHER, or NO_CHANGE")
    
    class Config:
        json_schema_extra = {
            "example": {
                "has_recommendation": True,
                "profile_id": "TREND_DEFAULT",
                "suggested_scale_factor": 0.95,
                "reason": "Hit rate below target, bringing TPs closer",
                "confidence": 0.65,
                "direction": "CLOSER"
            }
        }


class TPDashboardEntry(BaseModel):
    """Complete TP dashboard entry for one strategy/symbol pair."""
    key: TPDashboardKey = Field(..., description="Strategy/symbol identifier")
    profile: TPDashboardProfile = Field(..., description="Current TP profile")
    metrics: TPDashboardMetrics = Field(..., description="Performance metrics")
    recommendation: TPDashboardRecommendation = Field(..., description="Optimization recommendation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "key": {"strategy_id": "RL_V3", "symbol": "BTCUSDT"},
                "profile": {"profile_id": "TREND_DEFAULT", "legs": [], "trailing_profile_id": None},
                "metrics": {"tp_hit_rate": 0.62, "tp_attempts": 45, "tp_hits": 28, "tp_misses": 17},
                "recommendation": {"has_recommendation": False}
            }
        }


class TPDashboardSummary(BaseModel):
    """Summary of best and worst performing TP configurations."""
    best: List[TPDashboardEntry] = Field(..., description="Top performing entries")
    worst: List[TPDashboardEntry] = Field(..., description="Worst performing entries")
    total_entries: int = Field(..., description="Total number of entries analyzed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "best": [],
                "worst": [],
                "total_entries": 10
            }
        }


# ============================================================================
# Service Class
# ============================================================================

class TPDashboardService:
    """
    Service facade for TP v3 dashboard analytics.
    
    Aggregates data from:
    - TPPerformanceTracker (metrics)
    - tp_profiles_v3 (profile configurations)
    - TPOptimizerV3 (recommendations)
    """
    
    def __init__(
        self,
        tp_tracker: Optional[TPPerformanceTracker] = None,
        regime: MarketRegime = MarketRegime.NORMAL
    ):
        """
        Initialize TP Dashboard Service.
        
        Args:
            tp_tracker: TPPerformanceTracker instance (default: singleton)
            regime: Market regime for profile lookup (default: NORMAL)
        """
        self.logger = logging.getLogger(__name__)
        self.tp_tracker = tp_tracker or get_tp_tracker()
        self.regime = regime
        self.optimizer = TPOptimizerV3()
    
    def list_tp_entities(self) -> List[TPDashboardKey]:
        """
        List all strategy/symbol pairs with TP metrics.
        
        Returns:
            List of TPDashboardKey for tracked pairs
        """
        all_metrics = self.tp_tracker.get_metrics()
        
        entities = [
            TPDashboardKey(
                strategy_id=metrics.strategy_id,
                symbol=metrics.symbol
            )
            for metrics in all_metrics
        ]
        
        self.logger.info(f"[TP Dashboard Service] Found {len(entities)} tracked pairs")
        return entities
    
    def get_tp_dashboard_entry(
        self,
        strategy_id: str,
        symbol: str
    ) -> Optional[TPDashboardEntry]:
        """
        Get complete dashboard entry for a strategy/symbol pair.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            
        Returns:
            TPDashboardEntry with metrics, profile, and recommendation
            or None if no metrics exist
        """
        # Get metrics
        metrics_list = self.tp_tracker.get_metrics(
            strategy_id=strategy_id,
            symbol=symbol
        )
        
        if not metrics_list:
            self.logger.warning(
                f"[TP Dashboard Service] No metrics for {strategy_id}/{symbol}"
            )
            return None
        
        metrics = metrics_list[0]
        
        # Build entry components
        key = TPDashboardKey(strategy_id=strategy_id, symbol=symbol)
        profile = self._build_profile(strategy_id, symbol)
        metrics_model = self._build_metrics(metrics)
        recommendation = self._build_recommendation(strategy_id, symbol)
        
        return TPDashboardEntry(
            key=key,
            profile=profile,
            metrics=metrics_model,
            recommendation=recommendation
        )
    
    def get_top_best_and_worst(self, limit: int = 10) -> TPDashboardSummary:
        """
        Get top best and worst performing TP configurations.
        
        Ranks by composite score considering:
        - TP hit rate
        - Average R multiple
        - Total profit
        
        Args:
            limit: Number of entries to return for best/worst
            
        Returns:
            TPDashboardSummary with best and worst entries
        """
        # Get all entries
        entities = self.list_tp_entities()
        entries = []
        
        for entity in entities:
            entry = self.get_tp_dashboard_entry(
                strategy_id=entity.strategy_id,
                symbol=entity.symbol
            )
            if entry:
                entries.append(entry)
        
        if not entries:
            self.logger.warning("[TP Dashboard Service] No entries for summary")
            return TPDashboardSummary(best=[], worst=[], total_entries=0)
        
        # Score each entry
        scored_entries = [
            (self._calculate_performance_score(entry), entry)
            for entry in entries
        ]
        
        # Sort by score (higher is better)
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        # Extract top best and worst
        best = [entry for _, entry in scored_entries[:limit]]
        worst = [entry for _, entry in scored_entries[-limit:]]
        worst.reverse()  # Worst-first order
        
        self.logger.info(
            f"[TP Dashboard Service] Generated summary: "
            f"{len(best)} best, {len(worst)} worst from {len(entries)} total"
        )
        
        return TPDashboardSummary(
            best=best,
            worst=worst,
            total_entries=len(entries)
        )
    
    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------
    
    def _build_profile(self, strategy_id: str, symbol: str) -> TPDashboardProfile:
        """Build TPDashboardProfile from tp_profiles_v3."""
        try:
            profile, trailing = get_tp_and_trailing_profile(
                symbol=symbol,
                strategy_id=strategy_id,
                regime=self.regime
            )
            
            # Convert legs
            legs = [
                TPDashboardProfileLeg(
                    label=f"TP{idx}",
                    r_multiple=leg.r_multiple,
                    size_fraction=leg.size_fraction,
                    kind=leg.kind.value
                )
                for idx, leg in enumerate(profile.tp_legs, start=1)
            ]
            
            # Trailing profile ID
            trailing_id = None
            if trailing:
                trailing_id = f"{profile.name}_TRAILING"
            
            return TPDashboardProfile(
                profile_id=profile.name,
                legs=legs,
                trailing_profile_id=trailing_id,
                description=profile.description
            )
            
        except Exception as e:
            self.logger.error(
                f"[TP Dashboard Service] Failed to get profile for "
                f"{strategy_id}/{symbol}: {e}"
            )
            # Return minimal profile
            return TPDashboardProfile(
                profile_id="UNKNOWN",
                legs=[],
                trailing_profile_id=None,
                description="Profile not available"
            )
    
    def _build_metrics(self, metrics: TPMetrics) -> TPDashboardMetrics:
        """Build TPDashboardMetrics from TPMetrics."""
        # Calculate avg R multiple
        avg_r = None
        if metrics.tp_hit_rate > 0:
            avg_r = min(1.0 / max(metrics.tp_hit_rate, 0.05), 5.0)
        
        return TPDashboardMetrics(
            tp_hit_rate=metrics.tp_hit_rate,
            tp_attempts=metrics.tp_attempts,
            tp_hits=metrics.tp_hits,
            tp_misses=metrics.tp_misses,
            avg_r_multiple=avg_r,
            avg_slippage_pct=metrics.avg_slippage_pct if metrics.avg_slippage_pct > 0 else None,
            max_slippage_pct=metrics.max_slippage_pct if metrics.max_slippage_pct > 0 else None,
            avg_time_to_tp_minutes=metrics.avg_time_to_tp_minutes if metrics.avg_time_to_tp_minutes > 0 else None,
            total_tp_profit_usd=metrics.total_tp_profit_usd if metrics.total_tp_profit_usd > 0 else None,
            avg_tp_profit_usd=metrics.avg_tp_profit_usd if metrics.avg_tp_profit_usd > 0 else None,
            premature_exits=metrics.premature_exits if metrics.premature_exits > 0 else None,
            missed_opportunities_usd=metrics.missed_opportunities_usd if metrics.missed_opportunities_usd > 0 else None,
            last_updated=metrics.last_updated
        )
    
    def _build_recommendation(
        self,
        strategy_id: str,
        symbol: str
    ) -> TPDashboardRecommendation:
        """Build TPDashboardRecommendation from TPOptimizerV3."""
        try:
            rec = self.optimizer.evaluate_profile(
                strategy_id=strategy_id,
                symbol=symbol,
                regime=self.regime
            )
            
            if rec:
                return TPDashboardRecommendation(
                    has_recommendation=True,
                    profile_id=rec.current_profile_id,
                    suggested_scale_factor=rec.suggested_scale_factor,
                    reason=rec.reason,
                    confidence=rec.confidence,
                    direction=rec.direction.value
                )
            
        except Exception as e:
            self.logger.warning(
                f"[TP Dashboard Service] Failed to get recommendation for "
                f"{strategy_id}/{symbol}: {e}"
            )
        
        # No recommendation
        return TPDashboardRecommendation(has_recommendation=False)
    
    def _calculate_performance_score(self, entry: TPDashboardEntry) -> float:
        """
        Calculate composite performance score for ranking.
        
        Score formula:
        - Hit rate: 40% weight (0.0-1.0)
        - Avg R multiple: 30% weight (normalized 0.0-1.0, cap at 3R)
        - Total profit: 30% weight (normalized by max profit)
        
        Args:
            entry: TPDashboardEntry to score
            
        Returns:
            Performance score (0.0-100.0, higher is better)
        """
        metrics = entry.metrics
        
        # Hit rate component (0-40 points)
        hit_rate_score = metrics.tp_hit_rate * 40.0
        
        # Avg R component (0-30 points)
        # Normalize R to 0-1 range (assuming 3R is excellent)
        r_score = 0.0
        if metrics.avg_r_multiple:
            normalized_r = min(metrics.avg_r_multiple / 3.0, 1.0)
            r_score = normalized_r * 30.0
        
        # Profit component (0-30 points)
        # This is harder to normalize without global context
        # Use logarithmic scale for profit
        profit_score = 0.0
        if metrics.total_tp_profit_usd and metrics.total_tp_profit_usd > 0:
            # Log scale: $100 = 10 pts, $1000 = 20 pts, $10000 = 30 pts
            import math
            profit_score = min(math.log10(metrics.total_tp_profit_usd) * 10.0, 30.0)
        
        total_score = hit_rate_score + r_score + profit_score
        
        return total_score


# ============================================================================
# Singleton Access
# ============================================================================

_tp_dashboard_service: Optional[TPDashboardService] = None


def get_tp_dashboard_service() -> TPDashboardService:
    """Get or create TP Dashboard Service singleton."""
    global _tp_dashboard_service
    if _tp_dashboard_service is None:
        _tp_dashboard_service = TPDashboardService()
    return _tp_dashboard_service

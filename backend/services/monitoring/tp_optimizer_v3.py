"""
TP Optimizer v3 - Profile Adjustment Engine
============================================

Analyzes TP performance metrics and generates profile adjustment recommendations.
Integrates with TPPerformanceTracker and TPProfile system to optimize take-profit
distances based on hit rates and R multiples.

Features:
- Rule-based optimization logic
- Configurable target bands for hit rate and avg R
- Profile-level scale factor recommendations
- Runtime override support (Redis/PolicyStore)
- Non-intrusive design (recommendations only)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from backend.services.monitoring.tp_performance_tracker import (
    TPPerformanceTracker,
    TPMetrics,
    get_tp_performance_tracker
)
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
    TPProfile,
    MarketRegime,
    get_tp_and_trailing_profile,
    register_custom_profile,
    get_profile_by_name
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Models
# ============================================================================

@dataclass
class TPOptimizationTarget:
    """
    Target performance metrics for TP optimization.
    
    Defines acceptable ranges for hit rate and average R multiple.
    When actual metrics fall outside these ranges, optimizer suggests adjustments.
    """
    strategy_id: str
    symbol: str = "*"  # "*" = applies to all symbols
    
    # Hit rate targets
    min_hit_rate: float = 0.45  # Below this: TPs too far
    max_hit_rate: float = 0.70  # Above this: TPs too close
    
    # R multiple targets (profit per winner)
    min_avg_r: float = 1.2      # Below this: profits too small
    max_avg_r: Optional[float] = None  # Above this: could take profit sooner
    
    # Minimum sample size before optimizing
    min_attempts: int = 20
    
    # Adjustment sensitivity
    adjustment_step: float = 0.05  # 5% adjustment increments


class AdjustmentDirection(str, Enum):
    """Direction to adjust TP distances"""
    CLOSER = "CLOSER"      # Bring TPs closer (scale < 1.0)
    FURTHER = "FURTHER"    # Push TPs further (scale > 1.0)
    NO_CHANGE = "NO_CHANGE"


@dataclass
class TPAdjustmentRecommendation:
    """
    Recommendation to adjust a TP profile.
    
    Contains all information needed to apply the adjustment,
    either manually or via automated override system.
    """
    strategy_id: str
    symbol: str
    profile_name: str
    current_profile_id: str  # Name of currently used profile
    
    # Adjustment parameters
    direction: AdjustmentDirection
    suggested_scale_factor: float  # Multiplier for r_multiples (0.9 = closer, 1.1 = further)
    
    # Rationale
    reason: str
    confidence: float = 0.0  # 0.0-1.0 confidence in recommendation
    
    # Supporting data
    metrics_snapshot: Dict = field(default_factory=dict)
    targets: Dict = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied: bool = False


# ============================================================================
# TP Optimizer v3
# ============================================================================

class TPOptimizerV3:
    """
    TP Profile Optimizer - Generates adjustment recommendations based on performance.
    
    Workflow:
    1. Load optimization targets (strategy_id, symbol, desired metrics)
    2. Fetch actual metrics from TPPerformanceTracker
    3. Compare actual vs target, generate recommendations
    4. Optionally apply recommendations via runtime overrides
    
    Does NOT automatically modify profiles - only provides recommendations.
    """
    
    def __init__(
        self,
        tp_tracker: Optional[TPPerformanceTracker] = None,
        targets: Optional[List[TPOptimizationTarget]] = None
    ):
        """
        Initialize TP Optimizer v3.
        
        Args:
            tp_tracker: TPPerformanceTracker instance (uses singleton if None)
            targets: List of optimization targets (uses defaults if None)
        """
        self.logger = logging.getLogger(__name__)
        self.tp_tracker = tp_tracker or get_tp_performance_tracker()
        
        # Load targets
        self._targets = targets or self._load_default_targets()
        
        # Runtime overrides (in-memory for now, could be Redis/PolicyStore)
        self._runtime_overrides: Dict[Tuple[str, str], float] = {}
        
        self.logger.info(
            f"[TP Optimizer] Initialized with {len(self._targets)} optimization targets"
        )
    
    def _load_default_targets(self) -> List[TPOptimizationTarget]:
        """Load default optimization targets."""
        # Default targets for common strategies
        return [
            # RL_V3 strategy targets
            TPOptimizationTarget(
                strategy_id="RL_V3",
                symbol="*",
                min_hit_rate=0.45,
                max_hit_rate=0.70,
                min_avg_r=1.2,
                min_attempts=20
            ),
            # Aggressive strategy (higher hit rate, lower R)
            TPOptimizationTarget(
                strategy_id="SCALP_V2",
                symbol="*",
                min_hit_rate=0.60,
                max_hit_rate=0.85,
                min_avg_r=0.8,
                min_attempts=30
            ),
            # Conservative strategy (lower hit rate, higher R)
            TPOptimizationTarget(
                strategy_id="TREND_FOLLOW",
                symbol="*",
                min_hit_rate=0.35,
                max_hit_rate=0.60,
                min_avg_r=1.8,
                min_attempts=15
            )
        ]
    
    def load_targets(self, targets: List[TPOptimizationTarget]):
        """
        Load custom optimization targets.
        
        Args:
            targets: List of TPOptimizationTarget configurations
        """
        self._targets = targets
        self.logger.info(f"[TP Optimizer] Loaded {len(targets)} custom targets")
    
    def _get_target_for_pair(
        self,
        strategy_id: str,
        symbol: str
    ) -> Optional[TPOptimizationTarget]:
        """
        Find matching target for (strategy_id, symbol).
        
        Matches with specificity:
        1. (strategy_id, symbol) - exact match
        2. (strategy_id, "*") - strategy default
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            
        Returns:
            Matching target or None
        """
        # Try exact match first
        for target in self._targets:
            if target.strategy_id == strategy_id and target.symbol == symbol:
                return target
        
        # Try strategy wildcard
        for target in self._targets:
            if target.strategy_id == strategy_id and target.symbol == "*":
                return target
        
        return None
    
    def _calculate_avg_r_multiple(
        self,
        metrics: TPMetrics,
        current_profile: TPProfile
    ) -> float:
        """
        Calculate average R multiple from metrics.
        
        Uses inverse relationship between hit rate and R:
        - Higher hit rate → TPs closer → lower R
        - Lower hit rate → TPs further → higher R
        
        Formula: R ≈ 1 / hit_rate
        Examples: 50% hit → 2R, 70% hit → 1.4R, 30% hit → 3.3R
        
        Args:
            metrics: TP performance metrics
            current_profile: Current profile being used
            
        Returns:
            Estimated average R multiple
        """
        if metrics.tp_hit_rate > 0:
            # Use inverse relationship: R ≈ 1 / hit_rate
            estimated_r = 1.0 / max(metrics.tp_hit_rate, 0.05)  # Avoid division by zero
            return min(estimated_r, 5.0)  # Cap at 5R for sanity
        
        return 1.5  # Default assumption if no hits
    
    def evaluate_profile(
        self,
        strategy_id: str,
        symbol: str,
        regime: Optional[MarketRegime] = None
    ) -> Optional[TPAdjustmentRecommendation]:
        """
        Evaluate profile performance and generate recommendation if needed.
        
        Logic:
        - Low hit rate + high avg R → TPs too far → suggest CLOSER
        - High hit rate + low avg R → TPs too close → suggest FURTHER
        - Within target band → NO_CHANGE (no recommendation)
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            regime: Market regime (optional, uses NORMAL if None)
            
        Returns:
            TPAdjustmentRecommendation if adjustment needed, None otherwise
        """
        # Get target for this pair
        target = self._get_target_for_pair(strategy_id, symbol)
        if not target:
            self.logger.debug(
                f"[TP Optimizer] No target defined for {strategy_id}/{symbol}"
            )
            return None
        
        # Get metrics
        metrics_list = self.tp_tracker.get_metrics(strategy_id=strategy_id, symbol=symbol)
        if not metrics_list:
            self.logger.debug(
                f"[TP Optimizer] No metrics available for {strategy_id}/{symbol}"
            )
            return None
        
        metrics = metrics_list[0]  # Should only be one match
        
        # Check minimum sample size
        total_attempts = metrics.tp_hits + metrics.tp_misses
        if total_attempts < target.min_attempts:
            self.logger.debug(
                f"[TP Optimizer] Insufficient data for {strategy_id}/{symbol} "
                f"({total_attempts}/{target.min_attempts} attempts)"
            )
            return None
        
        # Get current profile
        current_profile, _ = get_tp_and_trailing_profile(
            symbol=symbol,
            strategy_id=strategy_id,
            regime=regime or MarketRegime.NORMAL
        )
        
        # Calculate avg R multiple
        avg_r = self._calculate_avg_r_multiple(metrics, current_profile)
        
        # Evaluate against targets
        hit_rate = metrics.tp_hit_rate
        
        direction = AdjustmentDirection.NO_CHANGE
        scale_factor = 1.0
        reason = ""
        confidence = 0.0
        
        # Decision logic
        if hit_rate < target.min_hit_rate:
            # Hit rate too low
            if avg_r >= target.min_avg_r:
                # Good R but missing too often → TPs too far
                direction = AdjustmentDirection.CLOSER
                scale_factor = 1.0 - target.adjustment_step
                reason = (
                    f"Hit rate {hit_rate:.1%} below target {target.min_hit_rate:.1%}, "
                    f"but avg R {avg_r:.2f} acceptable. Bringing TPs closer."
                )
                # Confidence based on how far below target
                confidence = min((target.min_hit_rate - hit_rate) / target.min_hit_rate, 1.0)
            else:
                # Low hit rate AND low R → problem with strategy, not TPs
                reason = (
                    f"Both hit rate {hit_rate:.1%} and avg R {avg_r:.2f} below target. "
                    f"Strategy performance issue - no TP adjustment recommended."
                )
                return None
        
        elif hit_rate > target.max_hit_rate:
            # Hit rate too high
            if avg_r < target.min_avg_r:
                # High hit rate but low R → TPs too close
                direction = AdjustmentDirection.FURTHER
                scale_factor = 1.0 + target.adjustment_step
                reason = (
                    f"Hit rate {hit_rate:.1%} above target {target.max_hit_rate:.1%}, "
                    f"but avg R {avg_r:.2f} below {target.min_avg_r:.2f}. Pushing TPs further."
                )
                # Confidence based on how far above target
                confidence = min((hit_rate - target.max_hit_rate) / target.max_hit_rate, 1.0)
            else:
                # High hit rate AND good R → optimal, maybe room to push further
                if target.max_avg_r and avg_r < target.max_avg_r:
                    direction = AdjustmentDirection.FURTHER
                    scale_factor = 1.0 + (target.adjustment_step / 2)  # Smaller adjustment
                    reason = (
                        f"Hit rate {hit_rate:.1%} high and avg R {avg_r:.2f} good. "
                        f"Cautiously extending TPs for higher R."
                    )
                    confidence = 0.5  # Lower confidence for this case
                else:
                    # Optimal performance
                    return None
        else:
            # Hit rate within target band
            if avg_r < target.min_avg_r:
                # Good hit rate but R too low
                direction = AdjustmentDirection.FURTHER
                scale_factor = 1.0 + target.adjustment_step
                reason = (
                    f"Hit rate {hit_rate:.1%} acceptable, but avg R {avg_r:.2f} "
                    f"below {target.min_avg_r:.2f}. Extending TPs for better R."
                )
                confidence = 0.6
            else:
                # Optimal: within target band for both metrics
                return None
        
        # Build recommendation
        if direction == AdjustmentDirection.NO_CHANGE:
            return None
        
        recommendation = TPAdjustmentRecommendation(
            strategy_id=strategy_id,
            symbol=symbol,
            profile_name=f"{current_profile.name}_ADJUSTED",
            current_profile_id=current_profile.name,
            direction=direction,
            suggested_scale_factor=scale_factor,
            reason=reason,
            confidence=confidence,
            metrics_snapshot={
                'tp_hit_rate': hit_rate,
                'avg_r_multiple': avg_r,
                'tp_attempts': total_attempts,
                'tp_hits': metrics.tp_hits,
                'tp_misses': metrics.tp_misses,
                'avg_slippage_pct': metrics.avg_slippage_pct,
                'premature_exits': metrics.premature_exits
            },
            targets={
                'min_hit_rate': target.min_hit_rate,
                'max_hit_rate': target.max_hit_rate,
                'min_avg_r': target.min_avg_r
            }
        )
        
        self.logger.info(
            f"[TP Optimizer] Recommendation: {strategy_id}/{symbol} → "
            f"{direction.value} (scale={scale_factor:.2f}, conf={confidence:.1%})"
        )
        
        return recommendation
    
    def apply_recommendation(
        self,
        recommendation: TPAdjustmentRecommendation,
        persist: bool = False
    ) -> Optional[TPProfile]:
        """
        Apply a TP adjustment recommendation.
        
        Options:
        1. Log only (persist=False): Just log the recommendation
        2. Runtime override (persist=False): Apply in-memory override
        3. Persistent override (persist=True): Register custom profile
        
        Args:
            recommendation: Adjustment recommendation to apply
            persist: If True, register as custom profile. If False, runtime override only.
            
        Returns:
            Adjusted TPProfile if created, None if just logged
        """
        if recommendation.applied:
            self.logger.warning(
                f"[TP Optimizer] Recommendation already applied: "
                f"{recommendation.strategy_id}/{recommendation.symbol}"
            )
            return None
        
        # Get current profile
        current_profile = get_profile_by_name(recommendation.current_profile_id)
        if not current_profile:
            self.logger.error(
                f"[TP Optimizer] Profile not found: {recommendation.current_profile_id}"
            )
            return None
        
        # Create adjusted profile
        adjusted_profile = self._create_adjusted_profile(
            base_profile=current_profile,
            scale_factor=recommendation.suggested_scale_factor,
            new_name=recommendation.profile_name
        )
        
        if not persist:
            # Log only
            self.logger.info(
                f"[TP Optimizer] LOG ONLY - Recommended adjustment:\n"
                f"  Strategy: {recommendation.strategy_id}\n"
                f"  Symbol: {recommendation.symbol}\n"
                f"  Profile: {recommendation.current_profile_id} → {recommendation.profile_name}\n"
                f"  Direction: {recommendation.direction.value}\n"
                f"  Scale Factor: {recommendation.suggested_scale_factor:.3f}\n"
                f"  Reason: {recommendation.reason}\n"
                f"  Confidence: {recommendation.confidence:.1%}\n"
                f"  Metrics: hit_rate={recommendation.metrics_snapshot['tp_hit_rate']:.1%}, "
                f"avg_r={recommendation.metrics_snapshot['avg_r_multiple']:.2f}"
            )
            return adjusted_profile
        
        # Apply runtime override
        key = (recommendation.strategy_id, recommendation.symbol)
        self._runtime_overrides[key] = recommendation.suggested_scale_factor
        
        # Register custom profile
        from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import register_custom_profile
        register_custom_profile(
            profile=adjusted_profile,
            symbol=recommendation.symbol,
            strategy_id=recommendation.strategy_id,
            regime=MarketRegime.NORMAL  # Apply to all regimes for now
        )
        
        recommendation.applied = True
        
        self.logger.info(
            f"[TP Optimizer] Applied adjustment: {recommendation.strategy_id}/{recommendation.symbol} "
            f"→ {recommendation.profile_name} (scale={recommendation.suggested_scale_factor:.3f})"
        )
        
        return adjusted_profile
    
    def _create_adjusted_profile(
        self,
        base_profile: TPProfile,
        scale_factor: float,
        new_name: str
    ) -> TPProfile:
        """
        Create adjusted profile by scaling R multiples.
        
        Args:
            base_profile: Base profile to adjust
            scale_factor: Multiplier for r_multiples (0.9 = closer, 1.1 = further)
            new_name: Name for adjusted profile
            
        Returns:
            New TPProfile with adjusted R multiples
        """
        from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import TPProfile, TPProfileLeg
        
        # Scale TP legs
        adjusted_legs = []
        for leg in base_profile.tp_legs:
            adjusted_legs.append(
                TPProfileLeg(
                    r_multiple=leg.r_multiple * scale_factor,
                    size_fraction=leg.size_fraction,
                    kind=leg.kind
                )
            )
        
        # Scale trailing activation if present
        adjusted_trailing = None
        if base_profile.trailing:
            from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import TrailingProfile
            adjusted_trailing = TrailingProfile(
                callback_pct=base_profile.trailing.callback_pct,
                activation_r=base_profile.trailing.activation_r * scale_factor,
                tightening_curve=[
                    (r * scale_factor, callback)
                    for r, callback in base_profile.trailing.tightening_curve
                ]
            )
        
        return TPProfile(
            name=new_name,
            tp_legs=adjusted_legs,
            trailing=adjusted_trailing,
            description=f"Adjusted from {base_profile.name} (scale={scale_factor:.3f})"
        )
    
    def optimize_all_profiles_once(self) -> List[TPAdjustmentRecommendation]:
        """
        Evaluate all tracked (strategy_id, symbol) pairs and generate recommendations.
        
        This is the main entry point for batch optimization (e.g., nightly job).
        
        Returns:
            List of all recommendations generated
        """
        self.logger.info("[TP Optimizer] Starting batch optimization...")
        
        recommendations = []
        
        # Get all tracked pairs from TPPerformanceTracker
        all_metrics = self.tp_tracker.get_metrics()
        
        for metrics in all_metrics:
            rec = self.evaluate_profile(
                strategy_id=metrics.strategy_id,
                symbol=metrics.symbol
            )
            if rec:
                recommendations.append(rec)
        
        self.logger.info(
            f"[TP Optimizer] Batch optimization complete: "
            f"{len(recommendations)} recommendations generated from {len(all_metrics)} pairs"
        )
        
        return recommendations
    
    def get_runtime_override(
        self,
        strategy_id: str,
        symbol: str
    ) -> Optional[float]:
        """
        Get runtime scale factor override for (strategy_id, symbol).
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            
        Returns:
            Scale factor if override exists, None otherwise
        """
        key = (strategy_id, symbol)
        return self._runtime_overrides.get(key)
    
    def clear_runtime_overrides(self):
        """Clear all runtime overrides."""
        self._runtime_overrides.clear()
        self.logger.info("[TP Optimizer] Cleared all runtime overrides")


# ============================================================================
# Convenience Functions
# ============================================================================

# Global singleton
_tp_optimizer: Optional[TPOptimizerV3] = None


def get_tp_optimizer() -> TPOptimizerV3:
    """Get or create TP Optimizer v3 singleton."""
    global _tp_optimizer
    if _tp_optimizer is None:
        _tp_optimizer = TPOptimizerV3()
    return _tp_optimizer


def optimize_profiles_for_strategy(strategy_id: str) -> List[TPAdjustmentRecommendation]:
    """
    Convenience function: optimize all profiles for a specific strategy.
    
    Args:
        strategy_id: Strategy to optimize
        
    Returns:
        List of recommendations
    """
    optimizer = get_tp_optimizer()
    tracker = get_tp_performance_tracker()
    
    recommendations = []
    metrics_list = tracker.get_metrics(strategy_id=strategy_id)
    
    for metrics in metrics_list:
        rec = optimizer.evaluate_profile(
            strategy_id=metrics.strategy_id,
            symbol=metrics.symbol
        )
        if rec:
            recommendations.append(rec)
    
    return recommendations

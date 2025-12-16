"""
TP Dashboard Router

REST endpoints for TP analytics and performance visualization.
Aggregates data from:
- TPPerformanceTracker (metrics)
- tp_profiles_v3 (current profiles)
- TPOptimizerV3 (recommendations)

Endpoints are read-only and designed for dashboard consumption.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone

from backend.api.dashboard.tp_models import (
    TPLegInfo,
    TrailingInfo,
    TPProfileInfo,
    TPMetricsInfo,
    TPRecommendationInfo,
    TPDashboardRow,
    TPDashboardSummary
)
from backend.services.monitoring.tp_performance_tracker import (
    get_tp_tracker,
    TPMetrics
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

router = APIRouter(prefix="/api/dashboard/tp", tags=["tp-dashboard"])


# ============================================================================
# Helper Functions
# ============================================================================

def _convert_tp_profile_to_info(profile: TPProfile) -> TPProfileInfo:
    """
    Convert TPProfile to TPProfileInfo response model.
    
    Args:
        profile: TPProfile from tp_profiles_v3
        
    Returns:
        TPProfileInfo with frontend-ready structure
    """
    # Convert legs
    legs = []
    for idx, leg in enumerate(profile.tp_legs, start=1):
        legs.append(TPLegInfo(
            label=f"TP{idx}",
            r_multiple=leg.r_multiple,
            size_fraction=leg.size_fraction,
            kind=leg.kind.value
        ))
    
    # Convert trailing
    trailing_info = None
    if profile.trailing:
        trailing_info = TrailingInfo(
            callback_pct=profile.trailing.callback_pct,
            activation_r=profile.trailing.activation_r,
            tightening_curve=[
                {"r_threshold": r, "callback_pct": cb}
                for r, cb in profile.trailing.tightening_curve
            ]
        )
    
    return TPProfileInfo(
        profile_id=profile.name,
        legs=legs,
        trailing=trailing_info,
        description=profile.description
    )


def _convert_metrics_to_info(metrics: TPMetrics, avg_r_multiple: Optional[float] = None) -> TPMetricsInfo:
    """
    Convert TPMetrics to TPMetricsInfo response model.
    
    Args:
        metrics: TPMetrics from TPPerformanceTracker
        avg_r_multiple: Optional avg R (if None, will estimate from hit rate)
        
    Returns:
        TPMetricsInfo with all available data
    """
    # Calculate avg R if not provided
    if avg_r_multiple is None and metrics.tp_hit_rate > 0:
        # Estimate using inverse relationship: R ≈ 1 / hit_rate
        avg_r_multiple = min(1.0 / max(metrics.tp_hit_rate, 0.05), 5.0)
    
    return TPMetricsInfo(
        strategy_id=metrics.strategy_id,
        symbol=metrics.symbol,
        tp_hit_rate=metrics.tp_hit_rate,
        tp_attempts=metrics.tp_attempts,
        tp_hits=metrics.tp_hits,
        tp_misses=metrics.tp_misses,
        avg_r_multiple_winners=avg_r_multiple,
        avg_slippage_pct=metrics.avg_slippage_pct if metrics.avg_slippage_pct > 0 else None,
        max_slippage_pct=metrics.max_slippage_pct if metrics.max_slippage_pct > 0 else None,
        avg_time_to_tp_minutes=metrics.avg_time_to_tp_minutes if metrics.avg_time_to_tp_minutes > 0 else None,
        premature_exit_rate=(
            metrics.premature_exits / metrics.tp_attempts
            if metrics.tp_attempts > 0 else None
        ),
        total_tp_profit_usd=metrics.total_tp_profit_usd if metrics.total_tp_profit_usd > 0 else None,
        last_updated=metrics.last_updated
    )


def _convert_recommendation_to_info(rec: TPAdjustmentRecommendation) -> TPRecommendationInfo:
    """
    Convert TPAdjustmentRecommendation to TPRecommendationInfo response model.
    
    Args:
        rec: TPAdjustmentRecommendation from TPOptimizerV3
        
    Returns:
        TPRecommendationInfo with recommendation details
    """
    return TPRecommendationInfo(
        profile_id=rec.current_profile_id,
        suggested_scale_factor=rec.suggested_scale_factor,
        direction=rec.direction.value,
        reason=rec.reason,
        confidence=rec.confidence,
        metrics_snapshot=rec.metrics_snapshot
    )


def _build_dashboard_row(
    strategy_id: str,
    symbol: str,
    metrics: TPMetrics,
    regime: MarketRegime = MarketRegime.NORMAL
) -> TPDashboardRow:
    """
    Build a complete TPDashboardRow for a strategy/symbol pair.
    
    Args:
        strategy_id: Strategy identifier
        symbol: Trading symbol
        metrics: TPMetrics from tracker
        regime: Market regime for profile lookup
        
    Returns:
        TPDashboardRow with metrics, profile, and recommendation
    """
    # Convert metrics
    metrics_info = _convert_metrics_to_info(metrics)
    
    # Get current profile
    profile_info = None
    try:
        profile, _ = get_tp_and_trailing_profile(
            symbol=symbol,
            strategy_id=strategy_id,
            regime=regime
        )
        profile_info = _convert_tp_profile_to_info(profile)
    except Exception as e:
        logger.warning(
            f"[TP Dashboard] Failed to get profile for {strategy_id}/{symbol}: {e}"
        )
    
    # Get optimizer recommendation (if available)
    recommendation_info = None
    try:
        optimizer = TPOptimizerV3()
        recommendation = optimizer.evaluate_profile(
            strategy_id=strategy_id,
            symbol=symbol,
            regime=regime
        )
        if recommendation:
            recommendation_info = _convert_recommendation_to_info(recommendation)
    except Exception as e:
        logger.warning(
            f"[TP Dashboard] Failed to get recommendation for {strategy_id}/{symbol}: {e}"
        )
    
    return TPDashboardRow(
        strategy_id=strategy_id,
        symbol=symbol,
        regime=regime.value,
        metrics=metrics_info,
        profile=profile_info,
        recommendation=recommendation_info
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "/summary",
    response_model=TPDashboardSummary,
    summary="Get TP analytics summary for all tracked pairs",
    description="""
    Returns complete TP performance dashboard for all strategy/symbol pairs
    that have metrics tracked by TPPerformanceTracker.
    
    For each pair, includes:
    - Performance metrics (hit rate, R multiple, slippage, timing, profit)
    - Current TP profile configuration
    - TPOptimizer recommendations (if adjustment needed)
    
    Use this endpoint to populate the main TP analytics dashboard table.
    """
)
async def get_tp_summary() -> TPDashboardSummary:
    """
    Get TP analytics summary for all tracked pairs.
    
    Returns:
        TPDashboardSummary with all tracked strategy/symbol pairs
    """
    try:
        # Get TP tracker
        tracker = get_tp_tracker()
        
        # Get all metrics
        all_metrics = tracker.get_metrics()
        
        if not all_metrics:
            logger.info("[TP Dashboard] No metrics available yet")
            return TPDashboardSummary(
                rows=[],
                total_pairs=0,
                generated_at=datetime.now(timezone.utc)
            )
        
        # Build dashboard rows
        rows = []
        for metrics in all_metrics:
            try:
                # For now, assume NORMAL regime (could be enhanced to fetch from regime detector)
                row = _build_dashboard_row(
                    strategy_id=metrics.strategy_id,
                    symbol=metrics.symbol,
                    metrics=metrics,
                    regime=MarketRegime.NORMAL
                )
                rows.append(row)
            except Exception as e:
                logger.error(
                    f"[TP Dashboard] Failed to build row for "
                    f"{metrics.strategy_id}/{metrics.symbol}: {e}"
                )
                # Continue with other pairs
                continue
        
        logger.info(f"[TP Dashboard] Generated summary with {len(rows)} pairs")
        
        return TPDashboardSummary(
            rows=rows,
            total_pairs=len(rows),
            generated_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"[TP Dashboard] Error generating summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate TP summary: {str(e)}"
        )


@router.get(
    "/{strategy_id}/{symbol}",
    response_model=TPDashboardRow,
    summary="Get TP analytics for specific strategy/symbol pair",
    description="""
    Returns detailed TP performance analytics for a single strategy/symbol pair.
    
    Includes:
    - Performance metrics (hit rate, R multiple, slippage, timing, profit)
    - Current TP profile with legs and trailing configuration
    - TPOptimizer recommendation (if adjustment needed)
    
    Returns 404 if no metrics exist for the requested pair.
    """
)
async def get_tp_for_pair(strategy_id: str, symbol: str) -> TPDashboardRow:
    """
    Get TP analytics for a specific strategy/symbol pair.
    
    Args:
        strategy_id: Strategy identifier (e.g., 'RL_V3')
        symbol: Trading symbol (e.g., 'BTCUSDT')
        
    Returns:
        TPDashboardRow with complete analytics
        
    Raises:
        HTTPException: 404 if no metrics exist for pair
    """
    try:
        # Get TP tracker
        tracker = get_tp_tracker()
        
        # Get metrics for this pair
        metrics_list = tracker.get_metrics(strategy_id=strategy_id, symbol=symbol)
        
        if not metrics_list:
            logger.warning(
                f"[TP Dashboard] No metrics found for {strategy_id}/{symbol}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No TP metrics found for {strategy_id}/{symbol}"
            )
        
        metrics = metrics_list[0]  # Should only be one match
        
        # Build dashboard row (assume NORMAL regime)
        row = _build_dashboard_row(
            strategy_id=strategy_id,
            symbol=symbol,
            metrics=metrics,
            regime=MarketRegime.NORMAL
        )
        
        logger.info(f"[TP Dashboard] Retrieved analytics for {strategy_id}/{symbol}")
        
        return row
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[TP Dashboard] Error getting analytics for {strategy_id}/{symbol}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get TP analytics: {str(e)}"
        )

"""
Dashboard TP API Routes

REST endpoints for TP v3 analytics dashboard.
Uses TPDashboardService for data aggregation.

Endpoints:
- GET /api/dashboard/tp/entities - List tracked strategy/symbol pairs
- GET /api/dashboard/tp/entry - Get entry for specific pair
- GET /api/dashboard/tp/summary - Get best/worst performing pairs
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Query, status

from backend.services.dashboard.tp_dashboard_service import (
    TPDashboardService,
    TPDashboardKey,
    TPDashboardEntry,
    TPDashboardSummary,
    get_tp_dashboard_service
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard/tp", tags=["dashboard-tp"])


@router.get(
    "/entities",
    response_model=List[TPDashboardKey],
    summary="List tracked strategy/symbol pairs",
    description="""
    Returns list of all strategy/symbol pairs that have TP metrics tracked.
    
    Use this endpoint to populate dropdown filters or discovery UI.
    """
)
async def get_tp_entities() -> List[TPDashboardKey]:
    """
    List all tracked strategy/symbol pairs.
    
    Returns:
        List of TPDashboardKey with strategy_id and symbol
    """
    try:
        service = get_tp_dashboard_service()
        entities = service.list_tp_entities()
        
        logger.info(f"[Dashboard TP API] Retrieved {len(entities)} tracked pairs")
        
        return entities
        
    except Exception as e:
        logger.error(f"[Dashboard TP API] Error listing entities: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list TP entities: {str(e)}"
        )


@router.get(
    "/entry",
    response_model=TPDashboardEntry,
    summary="Get TP dashboard entry for specific pair",
    description="""
    Returns complete TP analytics for a strategy/symbol pair including:
    - Performance metrics (hit rate, R multiple, slippage, timing, profit)
    - Current TP profile configuration (legs, trailing)
    - Optimization recommendation (if available)
    
    Returns 404 if no metrics exist for the requested pair.
    """
)
async def get_tp_entry(
    strategy_id: str = Query(..., description="Strategy identifier (e.g., 'RL_V3')"),
    symbol: str = Query(..., description="Trading symbol (e.g., 'BTCUSDT')")
) -> TPDashboardEntry:
    """
    Get complete dashboard entry for a strategy/symbol pair.
    
    Args:
        strategy_id: Strategy identifier
        symbol: Trading symbol
        
    Returns:
        TPDashboardEntry with metrics, profile, and recommendation
        
    Raises:
        HTTPException: 404 if no metrics exist for pair
    """
    try:
        service = get_tp_dashboard_service()
        entry = service.get_tp_dashboard_entry(
            strategy_id=strategy_id,
            symbol=symbol
        )
        
        if not entry:
            logger.warning(
                f"[Dashboard TP API] No entry found for {strategy_id}/{symbol}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No TP metrics found for {strategy_id}/{symbol}"
            )
        
        logger.info(
            f"[Dashboard TP API] Retrieved entry for {strategy_id}/{symbol}"
        )
        
        return entry
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[Dashboard TP API] Error getting entry for {strategy_id}/{symbol}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get TP entry: {str(e)}"
        )


@router.get(
    "/summary",
    response_model=TPDashboardSummary,
    summary="Get best and worst performing TP configurations",
    description="""
    Returns ranked summary of TP performance across all tracked pairs.
    
    Rankings consider:
    - TP hit rate (40% weight)
    - Average R multiple (30% weight)
    - Total profit (30% weight)
    
    Use this endpoint to show:
    - Top performers (what's working well)
    - Bottom performers (what needs optimization)
    """
)
async def get_tp_summary(
    limit: int = Query(10, ge=1, le=50, description="Number of entries for best/worst lists")
) -> TPDashboardSummary:
    """
    Get best and worst performing TP configurations.
    
    Args:
        limit: Number of entries to return (1-50, default 10)
        
    Returns:
        TPDashboardSummary with best and worst entries
    """
    try:
        service = get_tp_dashboard_service()
        summary = service.get_top_best_and_worst(limit=limit)
        
        logger.info(
            f"[Dashboard TP API] Generated summary: "
            f"{len(summary.best)} best, {len(summary.worst)} worst from "
            f"{summary.total_entries} entries"
        )
        
        return summary
        
    except Exception as e:
        logger.error(
            f"[Dashboard TP API] Error generating summary: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate TP summary: {str(e)}"
        )

"""
Execution Service - REST API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from .models import (
    OrderRequest, OrderResponse,
    Position, PositionListResponse,
    Trade, TradeListResponse,
    ExecutionMetrics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/execution", tags=["execution"])


# Injected service instance (set in main.py)
_service = None


def set_service(service):
    """Inject the service instance."""
    global _service
    _service = service


@router.post("/order", response_model=OrderResponse, summary="Place manual order")
async def place_order(request: OrderRequest):
    """
    Place a manual order.
    
    Flow:
    - Validates with ExecutionSafetyGuard
    - Places order via SafeOrderExecutor
    - Saves to TradeStore
    - Publishes events (order.placed, trade.opened)
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    logger.info(f"[API] Manual order request: {request.symbol} {request.side} {request.quantity}")
    
    try:
        response = await _service.execute_order(request)
        return response
    except Exception as e:
        logger.error(f"[API] Order placement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions", response_model=PositionListResponse, summary="Get current positions")
async def get_positions():
    """
    Get all current open positions.
    
    Returns:
    - List of positions with PnL, margin, liquidation price
    - Aggregated metrics (total margin, total PnL)
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        positions = await _service.get_positions()
        
        total_margin_usd = sum(p.margin_usd for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        return PositionListResponse(
            positions=positions,
            total_positions=len(positions),
            total_margin_usd=total_margin_usd,
            total_unrealized_pnl=total_unrealized_pnl
        )
    except Exception as e:
        logger.error(f"[API] Failed to get positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades", response_model=TradeListResponse, summary="Get trade history")
async def get_trades(
    limit: int = Query(100, ge=1, le=1000, description="Max trades to return"),
    status: Optional[str] = Query(None, description="Filter by status (open/closed)")
):
    """
    Get trade history from TradeStore.
    
    Query params:
    - limit: Max trades to return (1-1000)
    - status: Filter by status ('open' or 'closed')
    
    Returns:
    - List of trades with entry/exit prices, PnL, metadata
    - Aggregated counts (total, open, closed)
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        trades = await _service.get_trades(limit=limit, status=status)
        
        open_count = sum(1 for t in trades if t.status == "OPEN")
        closed_count = len(trades) - open_count
        
        return TradeListResponse(
            trades=trades,
            total_trades=len(trades),
            open_trades=open_count,
            closed_trades=closed_count
        )
    except Exception as e:
        logger.error(f"[API] Failed to get trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{trade_id}", response_model=Trade, summary="Get specific trade")
async def get_trade(trade_id: str):
    """
    Get detailed information for a specific trade.
    
    Path params:
    - trade_id: Trade identifier
    
    Returns:
    - Full trade details including entry/exit prices, PnL, metadata
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        trade = await _service.get_trade(trade_id)
        
        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        
        return trade
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Failed to get trade {trade_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=ExecutionMetrics, summary="Get execution metrics")
async def get_metrics():
    """
    Get execution performance metrics.
    
    Returns:
    - Order counts (placed, filled, failed)
    - Success rate
    - Average latency
    - Average slippage
    - Rate limiter status
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # TODO: Implement metrics collection
        return ExecutionMetrics(
            orders_placed_total=0,
            orders_filled_total=0,
            orders_failed_total=0,
            success_rate_pct=0.0,
            avg_order_latency_ms=0.0,
            avg_slippage_pct=0.0,
            rate_limit_tokens_available=_service.rate_limiter.get_tokens_available() if _service.rate_limiter else 0,
            rate_limit_tokens_per_minute=1200
        )
    except Exception as e:
        logger.error(f"[API] Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

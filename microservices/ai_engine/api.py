"""
AI Engine Service - REST API Endpoints
"""
from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime, timezone

from .models import (
    SignalRequest, SignalResponse,
    ModelPerformanceMetrics, EnsembleMetrics,
    MetaStrategyMetrics, RLSizingMetrics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

# Injected service instance (set in main.py)
_service = None


def set_service(service):
    """Inject the service instance."""
    global _service
    _service = service


@router.post("/signal", response_model=SignalResponse, summary="Generate AI signal")
async def generate_signal(request: SignalRequest):
    """
    Generate AI trading signal for a symbol.
    
    Pipeline:
    - Ensemble inference
    - Meta-Strategy selection
    - RL Position Sizing
    
    Returns complete trade intent.
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    price_str = f"${request.price:.2f}" if request.price else "N/A"
    logger.info(f"[API] Signal request: {request.symbol} @ {price_str}")
    
    # If market data provided, update price history
    if request.price and request.price > 0:
        await _service.update_price_history(request.symbol, request.price, request.volume or 0)
    
    try:
        decision = await _service.generate_signal(symbol=request.symbol, current_price=request.price)
        
        if not decision:
            raise HTTPException(status_code=404, detail=f"No signal generated for {request.symbol}")
        
        return SignalResponse(
            symbol=decision.symbol,
            action=decision.side,
            confidence=decision.confidence,
            ensemble_confidence=decision.ensemble_confidence,
            strategy=decision.strategy,
            regime=decision.regime or "unknown",
            position_size_usd=decision.position_size_usd,
            leverage=decision.leverage,
            reasoning=f"Ensemble confidence {decision.ensemble_confidence:.2f}, strategy {decision.strategy}" if request.include_reasoning else None,
            timestamp=decision.timestamp
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Signal generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/ensemble", response_model=EnsembleMetrics, summary="Get ensemble metrics")
async def get_ensemble_metrics():
    """Get ensemble performance metrics."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # TODO: Implement metrics collection
    return EnsembleMetrics(
        total_signals=_service._signals_generated,
        consensus_rate=0.75,  # Placeholder
        avg_confidence=0.70,
        model_agreement={"xgb": 0.80, "lgbm": 0.75, "nhits": 0.85, "patchtst": 0.70},
        last_updated=datetime.now(timezone.utc).isoformat()
    )


@router.get("/metrics/meta-strategy", response_model=MetaStrategyMetrics, summary="Get meta-strategy metrics")
async def get_meta_strategy_metrics():
    """Get meta-strategy performance metrics."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # TODO: Implement metrics collection
    return MetaStrategyMetrics(
        total_decisions=0,
        exploration_rate=0.10,
        top_strategies=[],
        avg_q_value=0.0,
        last_updated=datetime.now(timezone.utc).isoformat()
    )


@router.get("/metrics/rl-sizing", response_model=RLSizingMetrics, summary="Get RL sizing metrics")
async def get_rl_sizing_metrics():
    """Get RL position sizing metrics."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # TODO: Implement metrics collection
    return RLSizingMetrics(
        total_decisions=0,
        exploration_rate=0.15,
        avg_position_size_usd=500.0,
        avg_leverage=10.0,
        avg_risk_pct=0.02,
        last_updated=datetime.now(timezone.utc).isoformat()
    )

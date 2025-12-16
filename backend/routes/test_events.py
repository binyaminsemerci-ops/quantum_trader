"""
Test Event Flow API Endpoints
Allows testing the event-driven trading flow by generating test signals.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import os

# Import event publishers and logger early to catch any import errors
try:
    from backend.events.publishers import publish_signal_generated
    from backend.core.logger import get_logger
    EVENTS_AVAILABLE = True
except ImportError as e:
    EVENTS_AVAILABLE = False
    import logging
    logging.warning(f"[TEST EVENTS] Event publishers not available: {e}")

router = APIRouter(prefix="/testevents", tags=["test"])


class GenerateSignalRequest(BaseModel):
    """Request model for generating a test signal."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: str = Field(..., description="Trade side: BUY or SELL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0.0-1.0)")
    timeframe: str = Field(default="1h", description="Timeframe (e.g., 1h, 4h)")
    model_version: str = Field(default="test_v1", description="Model version identifier")
    trace_id: Optional[str] = Field(default=None, description="Optional trace ID for tracking")


class PositionClosedRequest(BaseModel):
    """Request model for simulating a closed position."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    entry_price: float = Field(..., gt=0, description="Entry price")
    exit_price: float = Field(..., gt=0, description="Exit price")
    size_usd: float = Field(default=100.0, gt=0, description="Position size in USD")
    leverage: float = Field(default=5.0, gt=0, le=125, description="Leverage used")
    is_long: bool = Field(default=True, description="True if long position, False if short")
    pnl_usd: Optional[float] = Field(default=None, description="P&L in USD (calculated if not provided)")
    pnl_pct: Optional[float] = Field(default=None, description="P&L percentage (calculated if not provided)")
    duration_seconds: float = Field(default=3600.0, description="Position duration in seconds")
    exit_reason: str = Field(default="MANUAL", description="Exit reason: TP, SL, MANUAL, etc.")
    entry_confidence: float = Field(default=0.75, ge=0.0, le=1.0, description="Original signal confidence")
    model_version: str = Field(default="test_v1", description="Model version identifier")
    trace_id: Optional[str] = Field(default=None, description="Optional trace ID for tracking")


@router.post("/generate_signal")
async def generate_test_signal(
    request: Request,
    signal: GenerateSignalRequest,
):
    """
    Generate a test trading signal and publish it to the event bus.
    
    This triggers the complete event-driven trading flow:
    1. signal.generated → SignalSubscriber
    2. RiskGuard validation
    3. trade.execution_requested → TradeSubscriber
    4. trade.executed → PositionSubscriber
    5. position.opened
    
    Returns the trace_id for tracking the signal through the system.
    """
    try:
        if not EVENTS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Event system not available. Event publishers could not be imported."
            )
        
        logger = get_logger(__name__)
        
        # Validate side
        if signal.side.upper() not in ["BUY", "SELL"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid side '{signal.side}'. Must be BUY or SELL."
            )
        
        # Generate trace_id if not provided
        trace_id = signal.trace_id or f"test-{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"[TEST] Generating test signal",
            extra={
                "symbol": signal.symbol,
                "side": signal.side,
                "confidence": signal.confidence,
                "trace_id": trace_id,
            }
        )
        
        # Publish signal to event bus
        await publish_signal_generated(
            symbol=signal.symbol,
            side=signal.side.upper(),
            confidence=signal.confidence,
            timeframe=signal.timeframe,
            model_version=signal.model_version,
            trace_id=trace_id,
        )
        
        logger.info(
            f"[TEST] ✅ Test signal published successfully",
            extra={
                "trace_id": trace_id,
                "symbol": signal.symbol,
            }
        )
        
        return {
            "status": "success",
            "message": f"Test signal published for {signal.symbol}",
            "trace_id": trace_id,
            "symbol": signal.symbol,
            "side": signal.side.upper(),
            "confidence": signal.confidence,
            "timeframe": signal.timeframe,
            "model_version": signal.model_version,
            "next_steps": [
                "1. Check logs for 'signal.generated' event",
                "2. SignalSubscriber will validate with RiskGuard",
                "3. If approved, 'trade.execution_requested' will be published",
                "4. TradeSubscriber will simulate trade execution",
                "5. Follow trace_id in logs to see full flow",
            ]
        }
        
    except Exception as e:
        logger.error(
            f"[TEST] ❌ Failed to generate test signal: {e}",
            exc_info=True,
            extra={
                "symbol": signal.symbol,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate test signal: {str(e)}"
        )


@router.post("/position_closed")
async def simulate_position_closed(
    request: Request,
    position: PositionClosedRequest,
):
    """
    Simulate a position.closed event to test learning system integration.
    
    This directly publishes a position.closed event which triggers:
    1. RL Position Sizing Agent - learns optimal sizing
    2. RL Meta Strategy Agent - learns model selection
    3. Model Supervisor - tracks model performance
    4. Drift Detector - detects model drift
    5. CLM - orchestrates retraining
    
    This is useful for testing the learning systems without executing real trades.
    """
    try:
        if not EVENTS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Event system not available. Event publishers could not be imported."
            )
        
        from backend.events.publishers import publish_position_closed
        logger = get_logger(__name__)
        
        # Calculate P&L if not provided
        pnl_pct = position.pnl_pct
        pnl_usd = position.pnl_usd
        
        if pnl_pct is None:
            if position.is_long:
                pnl_pct = ((position.exit_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - position.exit_price) / position.entry_price) * 100
        
        if pnl_usd is None:
            pnl_usd = (position.size_usd * position.leverage) * (pnl_pct / 100)
        
        # Generate trace_id if not provided
        trace_id = position.trace_id or f"test-pos-{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"[TEST] Simulating position closed",
            extra={
                "symbol": position.symbol,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "trace_id": trace_id,
            }
        )
        
        # Publish position.closed event
        await publish_position_closed(
            symbol=position.symbol,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            size_usd=position.size_usd,
            leverage=position.leverage,
            is_long=position.is_long,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            duration_seconds=position.duration_seconds,
            exit_reason=position.exit_reason,
            trace_id=trace_id,
            entry_confidence=position.entry_confidence,
            model_version=position.model_version,
        )
        
        logger.info(
            f"[TEST] ✅ Position closed event published successfully",
            extra={
                "trace_id": trace_id,
                "symbol": position.symbol,
                "pnl_usd": pnl_usd,
            }
        )
        
        return {
            "status": "success",
            "message": f"Position closed event published for {position.symbol}",
            "trace_id": trace_id,
            "position": {
                "symbol": position.symbol,
                "entry_price": position.entry_price,
                "exit_price": position.exit_price,
                "size_usd": position.size_usd,
                "leverage": position.leverage,
                "is_long": position.is_long,
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "duration_seconds": position.duration_seconds,
                "exit_reason": position.exit_reason,
                "entry_confidence": position.entry_confidence,
                "model_version": position.model_version,
            },
            "learning_systems_notified": [
                "RL Position Sizing Agent",
                "RL Meta Strategy Agent",
                "Model Supervisor",
                "Drift Detector",
                "Continuous Learning Manager (CLM)",
            ],
            "next_steps": [
                "1. Check logs for 'position.closed' event",
                "2. PositionSubscriber will feed data to learning systems",
                "3. RL agents will update their policies",
                "4. Model Supervisor will track performance",
                "5. CLM may trigger retraining if needed",
            ]
        }
        
    except Exception as e:
        logger.error(
            f"[TEST] ❌ Failed to simulate position closed: {e}",
            exc_info=True,
            extra={
                "symbol": position.symbol,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to simulate position closed: {str(e)}"
        )


@router.get("/event_flow_status")
async def get_event_flow_status(request: Request):
    """
    Get the status of the event-driven trading flow.
    
    Returns information about registered subscribers and event bus status.
    """
    try:
        if not EVENTS_AVAILABLE:
            return {
                "status": "error",
                "message": "Event system not available. Event publishers could not be imported."
            }
        
        logger = get_logger(__name__)
        
        # Check if event flow is enabled
        import os
        event_flow_enabled = os.getenv("QT_EVENT_FLOW_ENABLED", "true").lower() == "true"
        
        if not event_flow_enabled:
            return {
                "status": "disabled",
                "message": "Event-driven trading flow is disabled. Set QT_EVENT_FLOW_ENABLED=true to enable.",
            }
        
        # Get subscribers from app state
        signal_subscriber = getattr(request.app.state, 'signal_subscriber', None)
        trade_subscriber = getattr(request.app.state, 'trade_subscriber', None)
        position_subscriber = getattr(request.app.state, 'position_subscriber', None)
        risk_subscriber = getattr(request.app.state, 'risk_subscriber', None)
        error_subscriber = getattr(request.app.state, 'error_subscriber', None)
        
        subscribers_status = {
            "signal_subscriber": "✅ initialized" if signal_subscriber else "❌ not initialized",
            "trade_subscriber": "✅ initialized" if trade_subscriber else "❌ not initialized",
            "position_subscriber": "✅ initialized" if position_subscriber else "❌ not initialized",
            "risk_subscriber": "✅ initialized" if risk_subscriber else "❌ not initialized",
            "error_subscriber": "✅ initialized" if error_subscriber else "❌ not initialized",
        }
        
        all_initialized = all([
            signal_subscriber,
            trade_subscriber,
            position_subscriber,
            risk_subscriber,
            error_subscriber,
        ])
        
        return {
            "status": "enabled" if all_initialized else "partial",
            "message": "Event-driven trading flow is enabled",
            "subscribers": subscribers_status,
            "event_types": [
                "signal.generated",
                "trade.execution_requested",
                "trade.executed",
                "position.opened",
                "position.closed",
                "risk.alert",
                "system.event_error",
            ],
            "ready": all_initialized,
        }
        
    except Exception as e:
        logger.error(f"[TEST] Failed to get event flow status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get event flow status: {str(e)}"
        )

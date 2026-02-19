"""
Event Publishers
================

Production-ready publishers for emitting events to EventBus v2 (Redis Streams).

All publishers:
- Inject trace_id for distributed tracing
- Use Logger v2 (structlog) for structured logging
- Handle EventBus unavailability gracefully
- Return publish result for error handling

Usage:
    from backend.events.publishers import publish_signal_generated
    
    await publish_signal_generated(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.85,
        timeframe="1h",
        model_version="ensemble_v2",
        trace_id="abc-123"
    )

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone

from backend.events.event_types import EventType
from backend.events.schemas import (
    SignalGeneratedEvent,
    TradeExecutionRequestedEvent,
    TradeExecutedEvent,
    PositionOpenedEvent,
    PositionClosedEvent,
    RiskAlertEvent,
    SystemEventErrorEvent,
)
from backend.core.event_bus import get_event_bus
from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _publish_event(
    event_type: EventType,
    payload: Dict[str, Any],
    trace_id: str,
) -> bool:
    """
    Internal helper to publish events to EventBus.
    
    Args:
        event_type: Event type enum
        payload: Event payload dict
        trace_id: Distributed trace ID
        
    Returns:
        True if published successfully, False otherwise
    """
    try:
        event_bus = get_event_bus()
        if not event_bus:
            logger.error(
                "event_publish_failed_no_bus",
                event_type=str(event_type),
                trace_id=trace_id,
            )
            return False
        
        await event_bus.publish(str(event_type), payload)
        
        logger.info(
            "event_published",
            event_type=str(event_type),
            trace_id=trace_id,
            payload_keys=list(payload.keys()),
        )
        return True
        
    except Exception as e:
        logger.error(
            "event_publish_error",
            event_type=str(event_type),
            trace_id=trace_id,
            error=str(e),
            exc_info=True,
        )
        return False


# ============================================================================
# TRADING SIGNAL PUBLISHERS
# ============================================================================

async def publish_signal_generated(
    symbol: str,
    side: str,
    confidence: float,
    timeframe: str,
    model_version: str,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish signal.generated event.
    
    Called by: AI Trading Engine (ML models, RL agents)
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        side: "BUY" or "SELL"
        confidence: Model confidence (0-1)
        timeframe: Signal timeframe (e.g., "1h", "4h")
        model_version: Model identifier
        trace_id: Optional trace ID (generated if None)
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = SignalGeneratedEvent(
        symbol=symbol,
        side=side,
        confidence=confidence,
        timeframe=timeframe,
        model_version=model_version,
        trace_id=trace_id or SignalGeneratedEvent().trace_id,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.SIGNAL_GENERATED,
        event.model_dump(),
        event.trace_id,
    )


# ============================================================================
# TRADE EXECUTION PUBLISHERS
# ============================================================================

async def publish_execution_requested(
    symbol: str,
    side: str,
    leverage: float,
    position_size_usd: float,
    trade_risk_pct: float,
    confidence: float,
    trace_id: str,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish trade.execution_requested event.
    
    Called by: Signal Subscriber (after RiskGuard approval)
    
    Args:
        symbol: Trading pair
        side: "BUY" or "SELL"
        leverage: Leverage (1-125x)
        position_size_usd: Position size in USD
        trade_risk_pct: Risk % of account
        confidence: Signal confidence
        trace_id: Trace ID from signal
        stop_loss_pct: Optional stop loss %
        take_profit_pct: Optional take profit %
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = TradeExecutionRequestedEvent(
        symbol=symbol,
        side=side,
        leverage=leverage,
        position_size_usd=position_size_usd,
        trade_risk_pct=trade_risk_pct,
        confidence=confidence,
        trace_id=trace_id,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.TRADE_EXECUTION_REQUESTED,
        event.model_dump(),
        event.trace_id,
    )


async def publish_trade_executed(
    symbol: str,
    side: str,
    entry_price: float,
    position_size_usd: float,
    leverage: float,
    order_id: str,
    trace_id: str,
    commission_usd: Optional[float] = None,
    slippage_pct: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish trade.executed event.
    
    Called by: Execution Engine (after Binance order filled)
    
    Args:
        symbol: Trading pair
        side: "BUY" or "SELL"
        entry_price: Fill price
        position_size_usd: Position size in USD
        leverage: Applied leverage
        order_id: Exchange order ID
        trace_id: Trace ID from execution request
        commission_usd: Optional trading fee
        slippage_pct: Optional slippage %
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = TradeExecutedEvent(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        position_size_usd=position_size_usd,
        leverage=leverage,
        order_id=order_id,
        trace_id=trace_id,
        commission_usd=commission_usd,
        slippage_pct=slippage_pct,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.TRADE_EXECUTED,
        event.model_dump(),
        event.trace_id,
    )


# ============================================================================
# POSITION LIFECYCLE PUBLISHERS
# ============================================================================

async def publish_position_opened(
    symbol: str,
    entry_price: float,
    size_usd: float,
    leverage: float,
    is_long: bool,
    trace_id: str,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish position.opened event.
    
    Called by: Position Monitor (after confirming position active)
    
    Args:
        symbol: Trading pair
        entry_price: Position entry price
        size_usd: Position size in USD
        leverage: Position leverage
        is_long: True if long, False if short
        trace_id: Trace ID from trade execution
        stop_loss_price: Optional stop loss price
        take_profit_price: Optional take profit price
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = PositionOpenedEvent(
        symbol=symbol,
        entry_price=entry_price,
        size_usd=size_usd,
        leverage=leverage,
        is_long=is_long,
        trace_id=trace_id,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.POSITION_OPENED,
        event.model_dump(),
        event.trace_id,
    )


async def publish_position_closed(
    symbol: str,
    entry_price: float,
    exit_price: float,
    size_usd: float,
    leverage: float,
    is_long: bool,
    pnl_usd: float,
    pnl_pct: float,
    duration_seconds: float,
    exit_reason: str,
    trace_id: str,
    max_drawdown_pct: Optional[float] = None,
    entry_confidence: Optional[float] = None,
    model_version: Optional[str] = None,
    market_condition: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish position.closed event.
    
    Called by: Position Monitor (after exit execution)
    
    This is the CRITICAL event for learning systems (RL, CLM, Supervisor).
    
    Args:
        symbol: Trading pair
        entry_price: Position entry price
        exit_price: Position exit price
        size_usd: Position size in USD
        leverage: Position leverage
        is_long: True if long, False if short
        pnl_usd: Profit/Loss in USD
        pnl_pct: Profit/Loss in %
        duration_seconds: Position duration
        exit_reason: Why position closed (TP/SL/MANUAL/SIGNAL)
        trace_id: Trace ID from position open
        max_drawdown_pct: Optional max drawdown during position
        entry_confidence: Optional entry signal confidence
        model_version: Optional model that generated signal
        market_condition: Optional market state at entry
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = PositionClosedEvent(
        symbol=symbol,
        entry_price=entry_price,
        exit_price=exit_price,
        size_usd=size_usd,
        leverage=leverage,
        is_long=is_long,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        duration_seconds=duration_seconds,
        exit_reason=exit_reason,
        trace_id=trace_id,
        max_drawdown_pct=max_drawdown_pct,
        entry_confidence=entry_confidence,
        model_version=model_version,
        market_condition=market_condition,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.POSITION_CLOSED,
        event.model_dump(),
        event.trace_id,
    )


# ============================================================================
# RISK MANAGEMENT PUBLISHERS
# ============================================================================

async def publish_risk_alert(
    severity: str,
    alert_type: str,
    message: str,
    trace_id: str,
    current_drawdown_pct: Optional[float] = None,
    max_allowed_drawdown_pct: Optional[float] = None,
    open_positions_count: Optional[int] = None,
    max_positions: Optional[int] = None,
    action_taken: Optional[str] = None,
    risk_profile: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish risk.alert event.
    
    Called by: SafetyGovernor, RiskGuard
    
    Args:
        severity: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
        alert_type: Type of risk alert
        message: Human-readable alert message
        trace_id: Trace ID
        current_drawdown_pct: Optional current drawdown %
        max_allowed_drawdown_pct: Optional max allowed drawdown %
        open_positions_count: Optional current open positions
        max_positions: Optional max allowed positions
        action_taken: Optional automated action (BLOCK/PAUSE/STOP)
        risk_profile: Optional active risk profile
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = RiskAlertEvent(
        severity=severity,
        alert_type=alert_type,
        message=message,
        trace_id=trace_id,
        current_drawdown_pct=current_drawdown_pct,
        max_allowed_drawdown_pct=max_allowed_drawdown_pct,
        open_positions_count=open_positions_count,
        max_positions=max_positions,
        action_taken=action_taken,
        risk_profile=risk_profile,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.RISK_ALERT,
        event.model_dump(),
        event.trace_id,
    )


# ============================================================================
# SYSTEM EVENT PUBLISHERS
# ============================================================================

async def publish_event_error(
    error_type: str,
    error_message: str,
    component: str,
    trace_id: str,
    event_type: Optional[str] = None,
    stack_trace: Optional[str] = None,
    event_payload: Optional[Dict[str, Any]] = None,
    retry_count: int = 0,
    is_recoverable: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish system.event_error event.
    
    Called by: Any subscriber/handler that encounters an error
    
    Args:
        error_type: Error class name
        error_message: Error description
        component: Component where error occurred
        trace_id: Trace ID
        event_type: Optional event being processed when error occurred
        stack_trace: Optional full stack trace
        event_payload: Optional event data that caused error
        retry_count: Number of retries attempted
        is_recoverable: Can this error be retried
        metadata: Optional additional data
        
    Returns:
        True if published successfully
    """
    event = SystemEventErrorEvent(
        error_type=error_type,
        error_message=error_message,
        component=component,
        trace_id=trace_id,
        event_type=event_type,
        stack_trace=stack_trace,
        event_payload=event_payload,
        retry_count=retry_count,
        is_recoverable=is_recoverable,
        metadata=metadata or {},
    )
    
    return await _publish_event(
        EventType.SYSTEM_EVENT_ERROR,
        event.model_dump(),
        event.trace_id,
    )

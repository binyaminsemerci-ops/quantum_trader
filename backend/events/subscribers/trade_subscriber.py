"""
Trade Subscriber
================

Listens to: trade.execution_requested
Published by: Signal Subscriber (after RiskGuard approval)

Flow:
    1. Receive trade.execution_requested event
    2. Trigger Execution Engine to execute on Binance
    3. Await order fill confirmation
    4. Publish trade.executed event
    5. Handle errors → publish system.event_error

This is the SECOND step in the event-driven trading pipeline.

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import traceback
from typing import Dict, Any, Optional

from backend.events.event_types import EventType
from backend.events.schemas import TradeExecutionRequestedEvent
from backend.events.publishers import publish_trade_executed, publish_event_error
from backend.core.logger import get_logger

logger = get_logger(__name__)


class TradeSubscriber:
    """
    Subscriber for trade.execution_requested events.
    
    Responsibilities:
    - Execute trades on exchange (Binance)
    - Wait for order fill confirmation
    - Calculate commission and slippage
    - Publish trade.executed event
    """
    
    def __init__(self, execution_engine=None):
        """
        Initialize trade subscriber.
        
        Args:
            execution_engine: ExecutionEngine instance for placing orders
        """
        self.execution_engine = execution_engine
        
        logger.info(
            "trade_subscriber_initialized",
            execution_engine_available=execution_engine is not None,
        )
    
    async def handle_execution_request(self, event_data: Dict[str, Any]) -> None:
        """
        Handle trade.execution_requested event.
        
        Args:
            event_data: Raw event payload from EventBus
        """
        try:
            # Deserialize event
            trade_request = TradeExecutionRequestedEvent(**event_data)
            
            logger.info(
                "execution_request_received",
                trace_id=trade_request.trace_id,
                symbol=trade_request.symbol,
                side=trade_request.side,
                leverage=trade_request.leverage,
                position_size_usd=trade_request.position_size_usd,
                confidence=trade_request.confidence,
            )
            
            # ================================================================
            # STEP 1: Validate execution engine availability
            # ================================================================
            
            if not self.execution_engine:
                logger.error(
                    "execution_engine_unavailable",
                    trace_id=trade_request.trace_id,
                    symbol=trade_request.symbol,
                )
                
                await publish_event_error(
                    error_type="ExecutionEngineUnavailable",
                    error_message="Execution engine not initialized",
                    component="TradeSubscriber",
                    trace_id=trade_request.trace_id,
                    event_type=str(EventType.TRADE_EXECUTION_REQUESTED),
                    event_payload=event_data,
                    is_recoverable=False,
                )
                return
            
            # ================================================================
            # STEP 2: Execute trade on exchange
            # ================================================================
            
            try:
                # Execute market order
                # TODO: Replace with actual execution engine call
                # For now, simulate execution
                
                logger.info(
                    "executing_trade",
                    trace_id=trade_request.trace_id,
                    symbol=trade_request.symbol,
                    side=trade_request.side,
                    leverage=trade_request.leverage,
                    position_size_usd=trade_request.position_size_usd,
                )
                
                # Simulated execution result
                # In production, this would come from ExecutionEngine
                execution_result = {
                    "order_id": f"ORDER_{trade_request.symbol}_{int(trade_request.timestamp)}",
                    "entry_price": 40000.0,  # TODO: Get actual fill price
                    "commission_usd": trade_request.position_size_usd * 0.0004,  # 0.04% fee
                    "slippage_pct": 0.02,  # 0.02% slippage
                    "status": "FILLED",
                }
                
                logger.info(
                    "trade_executed_on_exchange",
                    trace_id=trade_request.trace_id,
                    symbol=trade_request.symbol,
                    order_id=execution_result["order_id"],
                    entry_price=execution_result["entry_price"],
                    commission_usd=execution_result["commission_usd"],
                    slippage_pct=execution_result["slippage_pct"],
                )
                
            except Exception as e:
                logger.error(
                    "trade_execution_failed",
                    trace_id=trade_request.trace_id,
                    symbol=trade_request.symbol,
                    error=str(e),
                    exc_info=True,
                )
                
                await publish_event_error(
                    error_type=type(e).__name__,
                    error_message=f"Trade execution failed: {str(e)}",
                    component="TradeSubscriber.execution",
                    trace_id=trade_request.trace_id,
                    event_type=str(EventType.TRADE_EXECUTION_REQUESTED),
                    stack_trace=traceback.format_exc(),
                    event_payload=event_data,
                    is_recoverable=True,
                )
                return
            
            # ================================================================
            # STEP 3: Publish trade.executed event
            # ================================================================
            
            success = await publish_trade_executed(
                symbol=trade_request.symbol,
                side=trade_request.side,
                entry_price=execution_result["entry_price"],
                position_size_usd=trade_request.position_size_usd,
                leverage=trade_request.leverage,
                order_id=execution_result["order_id"],
                trace_id=trade_request.trace_id,
                commission_usd=execution_result["commission_usd"],
                slippage_pct=execution_result["slippage_pct"],
                metadata={
                    "confidence": trade_request.confidence,
                    "trade_risk_pct": trade_request.trade_risk_pct,
                    "stop_loss_pct": trade_request.stop_loss_pct,
                    "take_profit_pct": trade_request.take_profit_pct,
                    "original_metadata": trade_request.metadata,
                },
            )
            
            if success:
                logger.info(
                    "trade_executed_event_published",
                    trace_id=trade_request.trace_id,
                    symbol=trade_request.symbol,
                    order_id=execution_result["order_id"],
                    entry_price=execution_result["entry_price"],
                )
            else:
                logger.error(
                    "trade_executed_event_publish_failed",
                    trace_id=trade_request.trace_id,
                    symbol=trade_request.symbol,
                    order_id=execution_result["order_id"],
                )
        
        except Exception as e:
            trace_id = event_data.get("trace_id", "unknown")
            logger.error(
                "trade_handler_error",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            
            await publish_event_error(
                error_type=type(e).__name__,
                error_message=str(e),
                component="TradeSubscriber",
                trace_id=trace_id,
                event_type=str(EventType.TRADE_EXECUTION_REQUESTED),
                stack_trace=traceback.format_exc(),
                event_payload=event_data,
            )

"""
Position Subscriber
===================

Listens to: trade.executed, position.closed
Published by: Trade Subscriber, Position Monitor

Flow for trade.executed:
    1. Receive trade.executed event
    2. Confirm position is active
    3. Publish position.opened event

Flow for position.closed:
    1. Receive position.closed event
    2. Feed data to RL Position Sizing Agent
    3. Feed data to RL Meta Strategy Agent
    4. Feed data to Model Supervisor
    5. Feed data to Drift Detector
    6. Feed data to CLM

This is the THIRD step in the event-driven trading pipeline
and the PRIMARY integration point for learning systems.

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import traceback
from typing import Dict, Any, Optional

from backend.events.event_types import EventType
from backend.events.schemas import TradeExecutedEvent, PositionClosedEvent
from backend.events.publishers import (
    publish_position_opened,
    publish_event_error,
)
from backend.core.logger import get_logger

logger = get_logger(__name__)


class PositionSubscriber:
    """
    Subscriber for position lifecycle events.
    
    Responsibilities:
    - Confirm position opened after trade execution
    - Process closed positions for learning systems
    - Feed data to RL agents, CLM, Supervisor, Drift Detector
    """
    
    def __init__(
        self,
        rl_position_sizing=None,
        rl_meta_strategy=None,
        model_supervisor=None,
        drift_detector=None,
        clm=None,
    ):
        """
        Initialize position subscriber.
        
        Args:
            rl_position_sizing: RL Position Sizing Agent
            rl_meta_strategy: RL Meta Strategy Agent
            model_supervisor: Model Supervisor
            drift_detector: Drift Detector
            clm: Continuous Learning Manager
        """
        self.rl_position_sizing = rl_position_sizing
        self.rl_meta_strategy = rl_meta_strategy
        self.model_supervisor = model_supervisor
        self.drift_detector = drift_detector
        self.clm = clm
        
        logger.info(
            "position_subscriber_initialized",
            rl_position_sizing_available=rl_position_sizing is not None,
            rl_meta_strategy_available=rl_meta_strategy is not None,
            model_supervisor_available=model_supervisor is not None,
            drift_detector_available=drift_detector is not None,
            clm_available=clm is not None,
        )
    
    async def handle_trade_executed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle trade.executed event.
        
        Args:
            event_data: Raw event payload from EventBus
        """
        try:
            # Deserialize event
            trade = TradeExecutedEvent(**event_data)
            
            logger.info(
                "trade_executed_received",
                trace_id=trade.trace_id,
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                position_size_usd=trade.position_size_usd,
                order_id=trade.order_id,
            )
            
            # ================================================================
            # STEP 1: Confirm position is active
            # ================================================================
            # TODO: Query exchange to confirm position exists
            # For now, we assume trade.executed means position is open
            
            is_long = trade.side == "BUY"
            
            logger.info(
                "position_confirmed_active",
                trace_id=trade.trace_id,
                symbol=trade.symbol,
                entry_price=trade.entry_price,
                size_usd=trade.position_size_usd,
                is_long=is_long,
            )
            
            # ================================================================
            # STEP 2: Publish position.opened event
            # ================================================================
            
            success = await publish_position_opened(
                symbol=trade.symbol,
                entry_price=trade.entry_price,
                size_usd=trade.position_size_usd,
                leverage=trade.leverage,
                is_long=is_long,
                trace_id=trade.trace_id,
                metadata={
                    "order_id": trade.order_id,
                    "commission_usd": trade.commission_usd,
                    "slippage_pct": trade.slippage_pct,
                    "original_metadata": trade.metadata,
                },
            )
            
            if success:
                logger.info(
                    "position_opened_event_published",
                    trace_id=trade.trace_id,
                    symbol=trade.symbol,
                    entry_price=trade.entry_price,
                )
            else:
                logger.error(
                    "position_opened_event_publish_failed",
                    trace_id=trade.trace_id,
                    symbol=trade.symbol,
                )
        
        except Exception as e:
            trace_id = event_data.get("trace_id", "unknown")
            logger.error(
                "trade_executed_handler_error",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            
            await publish_event_error(
                error_type=type(e).__name__,
                error_message=str(e),
                component="PositionSubscriber.trade_executed",
                trace_id=trace_id,
                event_type=str(EventType.TRADE_EXECUTED),
                stack_trace=traceback.format_exc(),
                event_payload=event_data,
            )
    
    async def handle_position_closed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position.closed event.
        
        This is the CRITICAL event for learning systems.
        
        Args:
            event_data: Raw event payload from EventBus
        """
        try:
            # Deserialize event
            position = PositionClosedEvent(**event_data)
            
            logger.info(
                "position_closed_received",
                trace_id=position.trace_id,
                symbol=position.symbol,
                entry_price=position.entry_price,
                exit_price=position.exit_price,
                pnl_usd=position.pnl_usd,
                pnl_pct=position.pnl_pct,
                duration_seconds=position.duration_seconds,
                exit_reason=position.exit_reason,
            )
            
            # ================================================================
            # STEP 1: Feed to RL Position Sizing Agent
            # ================================================================
            
            if self.rl_position_sizing:
                try:
                    logger.info(
                        "feeding_rl_position_sizing",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                        pnl_pct=position.pnl_pct,
                    )
                    
                    # TODO: Implement actual RL agent feedback
                    # await self.rl_position_sizing.observe_outcome(
                    #     symbol=position.symbol,
                    #     size_usd=position.size_usd,
                    #     leverage=position.leverage,
                    #     pnl_pct=position.pnl_pct,
                    #     duration_seconds=position.duration_seconds,
                    # )
                    
                    logger.info(
                        "rl_position_sizing_fed",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                    )
                    
                except Exception as e:
                    logger.error(
                        "rl_position_sizing_feed_error",
                        trace_id=position.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # STEP 2: Feed to RL Meta Strategy Agent
            # ================================================================
            
            if self.rl_meta_strategy:
                try:
                    logger.info(
                        "feeding_rl_meta_strategy",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                        model_version=position.model_version,
                        pnl_pct=position.pnl_pct,
                    )
                    
                    # TODO: Implement actual RL agent feedback
                    # await self.rl_meta_strategy.observe_strategy_outcome(
                    #     model_version=position.model_version,
                    #     symbol=position.symbol,
                    #     confidence=position.entry_confidence,
                    #     pnl_pct=position.pnl_pct,
                    #     market_condition=position.market_condition,
                    # )
                    
                    logger.info(
                        "rl_meta_strategy_fed",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                    )
                    
                except Exception as e:
                    logger.error(
                        "rl_meta_strategy_feed_error",
                        trace_id=position.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # STEP 3: Feed to Model Supervisor
            # ================================================================
            
            if self.model_supervisor:
                try:
                    logger.info(
                        "feeding_model_supervisor",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                        model_version=position.model_version,
                        pnl_pct=position.pnl_pct,
                    )
                    
                    # TODO: Implement actual model supervisor feedback
                    # await self.model_supervisor.record_model_outcome(
                    #     model_id=position.model_version,
                    #     prediction_confidence=position.entry_confidence,
                    #     actual_outcome=position.pnl_pct > 0,
                    #     pnl_pct=position.pnl_pct,
                    # )
                    
                    logger.info(
                        "model_supervisor_fed",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                    )
                    
                except Exception as e:
                    logger.error(
                        "model_supervisor_feed_error",
                        trace_id=position.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # STEP 4: Feed to Drift Detector
            # ================================================================
            
            if self.drift_detector:
                try:
                    logger.info(
                        "feeding_drift_detector",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                        model_version=position.model_version,
                    )
                    
                    # TODO: Implement actual drift detection
                    # await self.drift_detector.observe_prediction(
                    #     model_id=position.model_version,
                    #     prediction=position.entry_confidence,
                    #     actual_outcome=position.pnl_pct > 0,
                    #     timestamp=position.timestamp,
                    # )
                    
                    logger.info(
                        "drift_detector_fed",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                    )
                    
                except Exception as e:
                    logger.error(
                        "drift_detector_feed_error",
                        trace_id=position.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # STEP 5: Feed to CLM (Continuous Learning Manager)
            # ================================================================
            
            if self.clm:
                try:
                    logger.info(
                        "feeding_clm",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                        pnl_pct=position.pnl_pct,
                    )
                    
                    # TODO: Implement actual CLM feedback
                    # await self.clm.record_trade_outcome(
                    #     symbol=position.symbol,
                    #     model_version=position.model_version,
                    #     entry_confidence=position.entry_confidence,
                    #     pnl_pct=position.pnl_pct,
                    #     duration_seconds=position.duration_seconds,
                    #     market_condition=position.market_condition,
                    # )
                    
                    logger.info(
                        "clm_fed",
                        trace_id=position.trace_id,
                        symbol=position.symbol,
                    )
                    
                except Exception as e:
                    logger.error(
                        "clm_feed_error",
                        trace_id=position.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # Summary log
            # ================================================================
            
            logger.info(
                "position_closed_processing_complete",
                trace_id=position.trace_id,
                symbol=position.symbol,
                pnl_usd=position.pnl_usd,
                pnl_pct=position.pnl_pct,
                learning_systems_fed=sum([
                    self.rl_position_sizing is not None,
                    self.rl_meta_strategy is not None,
                    self.model_supervisor is not None,
                    self.drift_detector is not None,
                    self.clm is not None,
                ]),
            )
        
        except Exception as e:
            trace_id = event_data.get("trace_id", "unknown")
            logger.error(
                "position_closed_handler_error",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            
            await publish_event_error(
                error_type=type(e).__name__,
                error_message=str(e),
                component="PositionSubscriber.position_closed",
                trace_id=trace_id,
                event_type=str(EventType.POSITION_CLOSED),
                stack_trace=traceback.format_exc(),
                event_payload=event_data,
            )

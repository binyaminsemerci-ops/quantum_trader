"""
Signal Subscriber
=================

Listens to: signal.generated
Published by: AI Trading Engine

Flow:
    1. Receive signal.generated event
    2. Validate signal quality
    3. Run RiskGuard.can_execute() with RiskProfile
    4. If approved → publish trade.execution_requested
    5. If denied → log denial reason
    6. Handle errors → publish system.event_error

This is the FIRST step in the event-driven trading pipeline.

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import traceback
from typing import Dict, Any, Optional

from backend.events.event_types import EventType
from backend.events.schemas import SignalGeneratedEvent
from backend.events.publishers import publish_execution_requested, publish_event_error
from backend.core.logger import get_logger

logger = get_logger(__name__)


class SignalSubscriber:
    """
    Subscriber for signal.generated events.
    
    Responsibilities:
    - Validate signal quality (confidence threshold)
    - Check risk limits via RiskGuard
    - Calculate position sizing
    - Publish approved trades for execution
    """
    
    def __init__(
        self,
        risk_guard=None,
        policy_store=None,
        min_confidence: float = 0.60,
    ):
        """
        Initialize signal subscriber.
        
        Args:
            risk_guard: RiskGuardService instance
            policy_store: PolicyStore instance
            min_confidence: Minimum signal confidence (default: 0.60)
        """
        self.risk_guard = risk_guard
        self.policy_store = policy_store
        self.min_confidence = min_confidence
        
        logger.info(
            "signal_subscriber_initialized",
            min_confidence=min_confidence,
            risk_guard_available=risk_guard is not None,
            policy_store_available=policy_store is not None,
        )
    
    async def handle_signal(self, event_data: Dict[str, Any]) -> None:
        """
        Handle signal.generated event.
        
        Args:
            event_data: Raw event payload from EventBus
        """
        try:
            # Deserialize event
            signal = SignalGeneratedEvent(**event_data)
            
            logger.info(
                "signal_received",
                trace_id=signal.trace_id,
                symbol=signal.symbol,
                side=signal.side,
                confidence=signal.confidence,
                model_version=signal.model_version,
            )
            
            # ================================================================
            # STEP 1: Validate signal quality
            # ================================================================
            
            # Get minimum confidence from RiskProfile if available
            min_confidence = self.min_confidence
            if self.policy_store:
                try:
                    risk_profile = await self.policy_store.get_active_risk_profile()
                    min_confidence = risk_profile.global_min_confidence
                    logger.debug(
                        "signal_min_confidence_from_policy",
                        trace_id=signal.trace_id,
                        profile_name=risk_profile.name,
                        min_confidence=min_confidence,
                    )
                except Exception as e:
                    logger.warning(
                        "signal_policy_read_failed",
                        trace_id=signal.trace_id,
                        error=str(e),
                        fallback_confidence=self.min_confidence,
                    )
            
            if signal.confidence < min_confidence:
                logger.warning(
                    "signal_rejected_low_confidence",
                    trace_id=signal.trace_id,
                    symbol=signal.symbol,
                    confidence=signal.confidence,
                    min_confidence=min_confidence,
                )
                return
            
            # ================================================================
            # STEP 2: Calculate position sizing
            # ================================================================
            
            # TODO: Integrate with RL Position Sizing Agent
            # For now, use basic position sizing from RiskProfile
            account_balance = 1000.0  # TODO: Get from account state
            
            if self.policy_store:
                try:
                    risk_profile = await self.policy_store.get_active_risk_profile()
                    leverage = risk_profile.max_leverage * 0.8  # Use 80% of max leverage
                    trade_risk_pct = risk_profile.max_risk_pct_per_trade
                    position_size_usd = min(
                        account_balance * (trade_risk_pct / 100) * leverage,
                        risk_profile.position_size_cap_usd
                    )
                except Exception as e:
                    logger.error(
                        "signal_position_sizing_error",
                        trace_id=signal.trace_id,
                        error=str(e),
                    )
                    # Fallback to conservative defaults
                    leverage = 3.0
                    trade_risk_pct = 1.0
                    position_size_usd = account_balance * 0.01 * leverage
            else:
                # No policy store - use conservative defaults
                leverage = 3.0
                trade_risk_pct = 1.0
                position_size_usd = account_balance * 0.01 * leverage
            
            logger.info(
                "signal_position_sized",
                trace_id=signal.trace_id,
                symbol=signal.symbol,
                leverage=leverage,
                position_size_usd=position_size_usd,
                trade_risk_pct=trade_risk_pct,
            )
            
            # ================================================================
            # STEP 3: Run RiskGuard checks
            # ================================================================
            
            if self.risk_guard:
                try:
                    allowed, denial_reason = await self.risk_guard.can_execute(
                        symbol=signal.symbol,
                        notional=position_size_usd,
                        leverage=leverage,
                        trade_risk_pct=trade_risk_pct,
                        position_size_usd=position_size_usd,
                        trace_id=signal.trace_id,
                    )
                    
                    if not allowed:
                        logger.warning(
                            "signal_rejected_by_risk_guard",
                            trace_id=signal.trace_id,
                            symbol=signal.symbol,
                            denial_reason=denial_reason,
                            leverage=leverage,
                            position_size_usd=position_size_usd,
                        )
                        return
                    
                    logger.info(
                        "signal_approved_by_risk_guard",
                        trace_id=signal.trace_id,
                        symbol=signal.symbol,
                        leverage=leverage,
                        position_size_usd=position_size_usd,
                    )
                    
                except Exception as e:
                    logger.error(
                        "signal_risk_guard_error",
                        trace_id=signal.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
                    await publish_event_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        component="SignalSubscriber.risk_guard",
                        trace_id=signal.trace_id,
                        event_type=str(EventType.SIGNAL_GENERATED),
                        stack_trace=traceback.format_exc(),
                        event_payload=event_data,
                    )
                    return
            else:
                logger.warning(
                    "signal_no_risk_guard",
                    trace_id=signal.trace_id,
                    message="RiskGuard not available - trade approved by default",
                )
            
            # ================================================================
            # STEP 4: Publish trade.execution_requested
            # ================================================================
            
            success = await publish_execution_requested(
                symbol=signal.symbol,
                side=signal.side,
                leverage=leverage,
                position_size_usd=position_size_usd,
                trade_risk_pct=trade_risk_pct,
                confidence=signal.confidence,
                trace_id=signal.trace_id,
                metadata={
                    "model_version": signal.model_version,
                    "timeframe": signal.timeframe,
                    "original_metadata": signal.metadata,
                },
            )
            
            if success:
                logger.info(
                    "signal_execution_requested",
                    trace_id=signal.trace_id,
                    symbol=signal.symbol,
                    side=signal.side,
                    leverage=leverage,
                    position_size_usd=position_size_usd,
                )
            else:
                logger.error(
                    "signal_execution_request_failed",
                    trace_id=signal.trace_id,
                    symbol=signal.symbol,
                )
        
        except Exception as e:
            trace_id = event_data.get("trace_id", "unknown")
            logger.error(
                "signal_handler_error",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            
            await publish_event_error(
                error_type=type(e).__name__,
                error_message=str(e),
                component="SignalSubscriber",
                trace_id=trace_id,
                event_type=str(EventType.SIGNAL_GENERATED),
                stack_trace=traceback.format_exc(),
                event_payload=event_data,
            )

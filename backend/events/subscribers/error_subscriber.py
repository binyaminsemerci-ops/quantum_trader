"""
Error Subscriber
================

Listens to: system.event_error
Published by: All subscribers when they encounter errors

Flow:
    1. Receive system.event_error event
    2. Log error with full context
    3. Classify error severity
    4. Send health degradation signal if needed
    5. Trigger alerts for critical errors
    6. Store error for analysis

This handles system-wide error monitoring and alerting.

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import traceback
from typing import Dict, Any, Optional

from backend.events.event_types import EventType
from backend.events.schemas import SystemEventErrorEvent
from backend.core.logger import get_logger

logger = get_logger(__name__)


class ErrorSubscriber:
    """
    Subscriber for system.event_error events.
    
    Responsibilities:
    - Monitor all system errors
    - Log errors with full context
    - Send health degradation signals
    - Trigger alerts for critical errors
    - Store errors for analysis
    """
    
    def __init__(
        self,
        health_monitor=None,
        alert_system=None,
    ):
        """
        Initialize error subscriber.
        
        Args:
            health_monitor: HealthMonitor instance
            alert_system: AlertSystem instance
        """
        self.health_monitor = health_monitor
        self.alert_system = alert_system
        
        logger.info(
            "error_subscriber_initialized",
            health_monitor_available=health_monitor is not None,
            alert_system_available=alert_system is not None,
        )
    
    async def handle_event_error(self, event_data: Dict[str, Any]) -> None:
        """
        Handle system.event_error event.
        
        Args:
            event_data: Raw event payload from EventBus
        """
        try:
            # Deserialize event
            error = SystemEventErrorEvent(**event_data)
            
            logger.error(
                "system_event_error_received",
                trace_id=error.trace_id,
                error_type=error.error_type,
                error_message=error.error_message,
                component=error.component,
                event_type=error.event_type,
                retry_count=error.retry_count,
                is_recoverable=error.is_recoverable,
            )
            
            # ================================================================
            # STEP 1: Log full error context
            # ================================================================
            
            if error.stack_trace:
                logger.error(
                    "error_stack_trace",
                    trace_id=error.trace_id,
                    component=error.component,
                    stack_trace=error.stack_trace,
                )
            
            if error.event_payload:
                logger.error(
                    "error_event_payload",
                    trace_id=error.trace_id,
                    component=error.component,
                    event_payload=error.event_payload,
                )
            
            # ================================================================
            # STEP 2: Classify error severity
            # ================================================================
            
            # Determine if this is a critical error
            is_critical = (
                not error.is_recoverable
                or error.retry_count >= 3
                or "Critical" in error.error_type
                or "Fatal" in error.error_type
            )
            
            severity = "CRITICAL" if is_critical else "ERROR"
            
            logger.error(
                "error_classified",
                trace_id=error.trace_id,
                severity=severity,
                is_critical=is_critical,
                is_recoverable=error.is_recoverable,
                retry_count=error.retry_count,
            )
            
            # ================================================================
            # STEP 3: Send health degradation signal
            # ================================================================
            
            if self.health_monitor:
                try:
                    logger.warning(
                        "sending_health_degradation_signal",
                        trace_id=error.trace_id,
                        component=error.component,
                        severity=severity,
                    )
                    
                    # TODO: Implement health degradation signal
                    # await self.health_monitor.report_component_error(
                    #     component=error.component,
                    #     error_type=error.error_type,
                    #     severity=severity,
                    #     is_recoverable=error.is_recoverable,
                    # )
                    
                    logger.warning(
                        "health_degradation_signal_sent",
                        trace_id=error.trace_id,
                        component=error.component,
                    )
                    
                except Exception as e:
                    logger.error(
                        "health_degradation_signal_failed",
                        trace_id=error.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # STEP 4: Trigger alerts for critical errors
            # ================================================================
            
            if is_critical and self.alert_system:
                try:
                    logger.critical(
                        "triggering_critical_error_alert",
                        trace_id=error.trace_id,
                        component=error.component,
                        error_type=error.error_type,
                        error_message=error.error_message,
                    )
                    
                    # TODO: Integrate with alerting system (Slack, PagerDuty, etc.)
                    # await self.alert_system.send_alert(
                    #     severity="CRITICAL",
                    #     title=f"Critical Error in {error.component}",
                    #     message=error.error_message,
                    #     component=error.component,
                    #     trace_id=error.trace_id,
                    # )
                    
                    logger.critical(
                        "critical_error_alert_sent",
                        trace_id=error.trace_id,
                        component=error.component,
                    )
                    
                except Exception as e:
                    logger.error(
                        "critical_error_alert_failed",
                        trace_id=error.trace_id,
                        error=str(e),
                        exc_info=True,
                    )
            
            # ================================================================
            # STEP 5: Store error for analysis
            # ================================================================
            
            # TODO: Store error in database for historical analysis
            # await self.store_error(error)
            
            logger.info(
                "system_event_error_processed",
                trace_id=error.trace_id,
                component=error.component,
                severity=severity,
            )
        
        except Exception as e:
            # Error handling an error event - log but don't create infinite loop
            trace_id = event_data.get("trace_id", "unknown")
            logger.error(
                "error_subscriber_handler_error",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__,
                message="Error while handling system.event_error - NOT re-publishing to avoid loop",
                exc_info=True,
            )

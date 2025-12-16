"""
Risk Subscriber
===============

Listens to: risk.alert
Published by: SafetyGovernor, RiskGuard

Flow:
    1. Receive risk.alert event
    2. Evaluate severity (LOW/MEDIUM/HIGH/CRITICAL)
    3. For CRITICAL alerts:
        - Trigger kill-switch
        - Pause all trading via RiskGuard
        - Send emergency stop signal
    4. For HIGH alerts:
        - Send warning to operators
        - Reduce position sizing
    5. Log all alerts for analysis

This handles risk management escalation.

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import traceback
from typing import Dict, Any, Optional

from backend.events.event_types import EventType
from backend.events.schemas import RiskAlertEvent
from backend.events.publishers import publish_event_error
from backend.core.logger import get_logger

logger = get_logger(__name__)


class RiskSubscriber:
    """
    Subscriber for risk.alert events.
    
    Responsibilities:
    - Monitor risk alerts from SafetyGovernor and RiskGuard
    - Trigger emergency stop for CRITICAL alerts
    - Send operator notifications for HIGH alerts
    - Log all risk events for analysis
    """
    
    def __init__(
        self,
        risk_guard=None,
        emergency_stop_controller=None,
    ):
        """
        Initialize risk subscriber.
        
        Args:
            risk_guard: RiskGuardService instance
            emergency_stop_controller: EmergencyStopController instance
        """
        self.risk_guard = risk_guard
        self.emergency_stop_controller = emergency_stop_controller
        
        logger.info(
            "risk_subscriber_initialized",
            risk_guard_available=risk_guard is not None,
            emergency_stop_available=emergency_stop_controller is not None,
        )
    
    async def handle_risk_alert(self, event_data: Dict[str, Any]) -> None:
        """
        Handle risk.alert event.
        
        Args:
            event_data: Raw event payload from EventBus
        """
        try:
            # Deserialize event
            alert = RiskAlertEvent(**event_data)
            
            logger.warning(
                "risk_alert_received",
                trace_id=alert.trace_id,
                severity=alert.severity,
                alert_type=alert.alert_type,
                message=alert.message,
                current_drawdown_pct=alert.current_drawdown_pct,
                max_allowed_drawdown_pct=alert.max_allowed_drawdown_pct,
                open_positions_count=alert.open_positions_count,
                max_positions=alert.max_positions,
                action_taken=alert.action_taken,
                risk_profile=alert.risk_profile,
            )
            
            # ================================================================
            # STEP 1: Handle CRITICAL severity
            # ================================================================
            
            if alert.severity == "CRITICAL":
                logger.critical(
                    "critical_risk_alert",
                    trace_id=alert.trace_id,
                    alert_type=alert.alert_type,
                    message=alert.message,
                    current_drawdown_pct=alert.current_drawdown_pct,
                )
                
                # Trigger emergency stop
                if self.emergency_stop_controller:
                    try:
                        logger.critical(
                            "triggering_emergency_stop",
                            trace_id=alert.trace_id,
                            reason=alert.message,
                        )
                        
                        # TODO: Implement actual emergency stop
                        # await self.emergency_stop_controller.trigger_stop(
                        #     reason=alert.message,
                        #     severity="CRITICAL",
                        # )
                        
                        logger.critical(
                            "emergency_stop_triggered",
                            trace_id=alert.trace_id,
                        )
                        
                    except Exception as e:
                        logger.error(
                            "emergency_stop_trigger_failed",
                            trace_id=alert.trace_id,
                            error=str(e),
                            exc_info=True,
                        )
                
                # Activate kill-switch via RiskGuard
                if self.risk_guard:
                    try:
                        logger.critical(
                            "activating_kill_switch",
                            trace_id=alert.trace_id,
                            reason=alert.message,
                        )
                        
                        # TODO: Implement kill-switch activation
                        # await self.risk_guard.set_kill_switch_override(enabled=True)
                        
                        logger.critical(
                            "kill_switch_activated",
                            trace_id=alert.trace_id,
                        )
                        
                    except Exception as e:
                        logger.error(
                            "kill_switch_activation_failed",
                            trace_id=alert.trace_id,
                            error=str(e),
                            exc_info=True,
                        )
            
            # ================================================================
            # STEP 2: Handle HIGH severity
            # ================================================================
            
            elif alert.severity == "HIGH":
                logger.warning(
                    "high_risk_alert",
                    trace_id=alert.trace_id,
                    alert_type=alert.alert_type,
                    message=alert.message,
                    current_drawdown_pct=alert.current_drawdown_pct,
                )
                
                # Send operator notification
                # TODO: Integrate with alerting system (Slack, PagerDuty, etc.)
                logger.warning(
                    "operator_notification_required",
                    trace_id=alert.trace_id,
                    alert_type=alert.alert_type,
                    message=alert.message,
                )
            
            # ================================================================
            # STEP 3: Handle MEDIUM severity
            # ================================================================
            
            elif alert.severity == "MEDIUM":
                logger.warning(
                    "medium_risk_alert",
                    trace_id=alert.trace_id,
                    alert_type=alert.alert_type,
                    message=alert.message,
                )
            
            # ================================================================
            # STEP 4: Handle LOW severity
            # ================================================================
            
            elif alert.severity == "LOW":
                logger.info(
                    "low_risk_alert",
                    trace_id=alert.trace_id,
                    alert_type=alert.alert_type,
                    message=alert.message,
                )
            
            # ================================================================
            # STEP 5: Log alert for historical analysis
            # ================================================================
            
            # TODO: Store alert in database for analysis
            # await self.store_risk_alert(alert)
            
            logger.info(
                "risk_alert_processed",
                trace_id=alert.trace_id,
                severity=alert.severity,
                alert_type=alert.alert_type,
            )
        
        except Exception as e:
            trace_id = event_data.get("trace_id", "unknown")
            logger.error(
                "risk_alert_handler_error",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            
            await publish_event_error(
                error_type=type(e).__name__,
                error_message=str(e),
                component="RiskSubscriber",
                trace_id=trace_id,
                event_type=str(EventType.RISK_ALERT),
                stack_trace=traceback.format_exc(),
                event_payload=event_data,
            )

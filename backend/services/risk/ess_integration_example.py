"""
Emergency Stop System (ESS) Integration Example

This demonstrates how to integrate ESS into the main Quantum Trader FastAPI backend.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from typing import AsyncGenerator

from backend.services.risk.emergency_stop_system import (
    EmergencyStopSystem,
    EmergencyStopController,
    DrawdownEmergencyEvaluator,
    SystemHealthEmergencyEvaluator,
    ExecutionErrorEmergencyEvaluator,
    DataFeedEmergencyEvaluator,
    ManualTriggerEmergencyEvaluator,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    FastAPI lifespan context manager.
    Starts ESS on startup, stops on shutdown.
    """
    logger.info("[STARTUP] Initializing Emergency Stop System...")
    
    # Get dependencies from app state
    policy_store = app.state.policy_store
    exchange_client = app.state.exchange_client
    event_bus = app.state.event_bus
    metrics_repo = app.state.metrics_repo
    health_monitor = app.state.health_monitor
    data_feed_monitor = app.state.data_feed_monitor
    
    # Create evaluators with production thresholds
    evaluators = [
        DrawdownEmergencyEvaluator(
            metrics_repo=metrics_repo,
            max_daily_loss_percent=float(os.getenv("QT_ESS_MAX_DAILY_LOSS", "10.0")),
            max_equity_drawdown_percent=float(os.getenv("QT_ESS_MAX_EQUITY_DD", "25.0")),
        ),
        SystemHealthEmergencyEvaluator(
            health_monitor=health_monitor,
        ),
        ExecutionErrorEmergencyEvaluator(
            metrics_repo=metrics_repo,
            max_sl_hits_per_period=int(os.getenv("QT_ESS_MAX_SL_HITS", "10")),
            period_minutes=int(os.getenv("QT_ESS_SL_PERIOD_MINUTES", "15")),
        ),
        DataFeedEmergencyEvaluator(
            data_feed_monitor=data_feed_monitor,
            max_staleness_minutes=int(os.getenv("QT_ESS_MAX_DATA_STALENESS", "5")),
        ),
        ManualTriggerEmergencyEvaluator(),
    ]
    
    # Create controller
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange_client,
        event_bus=event_bus,
    )
    
    # Create ESS
    ess = EmergencyStopSystem(
        evaluators=evaluators,
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=int(os.getenv("QT_ESS_CHECK_INTERVAL", "5")),
    )
    
    # Store in app state for API access
    app.state.ess = ess
    app.state.ess_controller = controller
    app.state.ess_manual_trigger = evaluators[-1]  # Manual trigger
    
    # Start ESS
    ess_task = ess.start()
    logger.info("🛡️ [ESS] Emergency Stop System ACTIVE")
    
    yield  # Application runs
    
    # Shutdown
    logger.info("[SHUTDOWN] Stopping Emergency Stop System...")
    await ess.stop()
    await ess_task
    logger.info("✅ [ESS] Emergency Stop System stopped")


# ============================================================================
# API ENDPOINTS
# ============================================================================

app = FastAPI(lifespan=lifespan)


def get_ess_controller(request: Request) -> EmergencyStopController:
    """Dependency to get ESS controller."""
    return request.app.state.ess_controller


def get_manual_trigger(request: Request) -> ManualTriggerEmergencyEvaluator:
    """Dependency to get manual trigger evaluator."""
    return request.app.state.ess_manual_trigger


@app.get("/api/emergency/status")
async def get_emergency_status(
    controller: EmergencyStopController = Depends(get_ess_controller)
):
    """
    Get current ESS status.
    
    Returns:
        - active: bool
        - status: str
        - reason: str | None
        - timestamp: str | None
        - activation_count: int
    """
    state = controller.state
    return {
        "active": state.active,
        "status": state.status.value,
        "reason": state.reason,
        "timestamp": state.timestamp.isoformat() if state.timestamp else None,
        "activation_count": state.activation_count,
    }


@app.post("/api/emergency/trigger")
async def trigger_emergency_stop(
    reason: str = "Manual admin trigger",
    manual_trigger: ManualTriggerEmergencyEvaluator = Depends(get_manual_trigger),
):
    """
    Manually trigger Emergency Stop System.
    
    ⚠️ WARNING: This will immediately close all positions and halt trading!
    
    Args:
        reason: Reason for emergency stop
    
    Returns:
        success: bool
        message: str
    """
    logger.warning(f"[API] Manual ESS trigger requested: {reason}")
    
    manual_trigger.trigger(reason)
    
    # Wait a moment for ESS to activate
    await asyncio.sleep(2)
    
    return {
        "success": True,
        "message": f"Emergency stop triggered: {reason}",
        "note": "ESS will activate within 5 seconds",
    }


@app.post("/api/emergency/reset")
async def reset_emergency_stop(
    reset_by: str = "admin",
    controller: EmergencyStopController = Depends(get_ess_controller),
):
    """
    Reset Emergency Stop System.
    
    ⚠️ Only call this after resolving the underlying issue!
    
    Args:
        reset_by: Who is resetting (for audit trail)
    
    Returns:
        success: bool
        message: str
    """
    if not controller.is_active:
        raise HTTPException(
            status_code=400,
            detail="ESS is not active, nothing to reset"
        )
    
    logger.warning(f"[API] ESS reset requested by: {reset_by}")
    
    await controller.reset(reset_by)
    
    return {
        "success": True,
        "message": f"Emergency stop reset by {reset_by}",
        "note": "Trading is now enabled",
    }


# ============================================================================
# ORCHESTRATOR INTEGRATION
# ============================================================================

class OrchestratorWithESS:
    """
    Example Orchestrator that checks ESS before executing trades.
    """
    
    def __init__(self, policy_store):
        self.policy_store = policy_store
    
    async def execute_signal(self, signal: dict) -> dict:
        """
        Execute a trading signal.
        
        Checks ESS status first - rejects all signals if ESS is active.
        """
        # Priority 1: Check ESS
        ess_state = self.policy_store.get("emergency_stop")
        if ess_state and ess_state.get("active"):
            logger.critical(
                f"[ORCHESTRATOR] ESS ACTIVE - Rejecting signal for {signal['symbol']}"
            )
            return {
                "success": False,
                "reason": f"ESS active: {ess_state['reason']}",
                "ess_activated": True,
            }
        
        # Normal execution...
        logger.info(f"[ORCHESTRATOR] Executing signal: {signal['symbol']}")
        
        # ... rest of orchestrator logic
        
        return {
            "success": True,
            "order_id": "12345",
        }


# ============================================================================
# RISK GUARD INTEGRATION
# ============================================================================

class RiskGuardWithESS:
    """
    Example Risk Guard that includes ESS validation.
    """
    
    def __init__(self, policy_store):
        self.policy_store = policy_store
    
    def validate_trade(self, trade: dict) -> dict:
        """
        Validate a trade before execution.
        
        ESS check is the FIRST validation - highest priority.
        """
        # Validation 1: ESS Status
        ess_state = self.policy_store.get("emergency_stop")
        if ess_state and ess_state.get("active"):
            return {
                "passed": False,
                "reason": f"ESS active: {ess_state['reason']}",
                "severity": "CRITICAL",
                "ess_block": True,
            }
        
        # Validation 2: Position limits
        # ... other risk checks
        
        return {
            "passed": True,
            "reason": "All validations passed",
        }


# ============================================================================
# EVENT HANDLER INTEGRATION
# ============================================================================

class EmergencyEventHandler:
    """
    Handles ESS events and triggers alerts.
    """
    
    def __init__(self, alert_service):
        self.alert_service = alert_service
    
    async def handle_emergency_stop(self, event):
        """
        Handle EmergencyStopEvent.
        
        Sends critical alerts to all channels.
        """
        logger.critical(f"[ALERT] ESS ACTIVATED: {event.reason}")
        
        # Send SMS/call to on-call engineer
        await self.alert_service.send_critical_alert(
            title="🚨 EMERGENCY STOP ACTIVATED",
            message=f"Reason: {event.reason}\n"
                   f"Positions closed: {event.positions_closed}\n"
                   f"Orders canceled: {event.orders_canceled}",
            priority="CRITICAL",
        )
        
        # Send Slack notification
        await self.alert_service.send_slack_message(
            channel="#trading-alerts",
            text=f"🚨 ESS ACTIVATED: {event.reason}",
            color="danger",
        )
        
        # Update dashboard
        await self.alert_service.update_dashboard_status("EMERGENCY_STOP")
        
        # Log to audit trail
        await self.alert_service.audit_log(
            event_type="ESS_ACTIVATION",
            severity="CRITICAL",
            data=event,
        )
    
    async def handle_emergency_reset(self, event):
        """
        Handle EmergencyResetEvent.
        """
        logger.warning(f"[ALERT] ESS RESET by {event.reset_by}")
        
        await self.alert_service.send_notification(
            title="✅ ESS Reset",
            message=f"Emergency stop reset by {event.reset_by}",
            priority="HIGH",
        )
        
        await self.alert_service.update_dashboard_status("OPERATIONAL")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_full_integration():
    """
    Example showing full ESS integration in production.
    """
    print("\n" + "="*80)
    print("ESS INTEGRATION EXAMPLE")
    print("="*80 + "\n")
    
    # This would be in your main FastAPI app startup
    
    # 1. ESS starts automatically via lifespan
    print("✅ ESS started via lifespan context")
    
    # 2. Orchestrator checks ESS before trades
    orchestrator = OrchestratorWithESS(policy_store)
    result = await orchestrator.execute_signal({
        "symbol": "BTCUSDT",
        "side": "BUY",
    })
    print(f"✅ Orchestrator result: {result}")
    
    # 3. Risk Guard validates ESS status
    risk_guard = RiskGuardWithESS(policy_store)
    validation = risk_guard.validate_trade({
        "symbol": "ETHUSDT",
        "size": 0.5,
    })
    print(f"✅ Risk Guard validation: {validation}")
    
    # 4. Event handlers process ESS events
    event_handler = EmergencyEventHandler(alert_service)
    # (Events published automatically by ESS)
    
    print("\n✅ Full integration complete")


if __name__ == "__main__":
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    # Note: In production, this would be started by FastAPI's lifespan
    print("See FastAPI app integration above")
    print("Run with: uvicorn backend.main:app")

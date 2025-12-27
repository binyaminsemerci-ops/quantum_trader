"""
System Health Monitor - Integration Guide for Quantum Trader

This file shows how to integrate the System Health Monitor into the main
Quantum Trader application (backend/main.py).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from backend.services.monitoring.system_health_monitor import (
    SystemHealthMonitor,
    HealthStatus,
    SystemHealthSummary,
    BaseHealthMonitor,
    HealthCheckResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Real Health Monitors (Production Implementations)
# ============================================================================

class MarketDataHealthMonitor(BaseHealthMonitor):
    """Monitor for Binance market data feed."""
    
    def __init__(self, market_data_client):
        super().__init__("market_data", heartbeat_timeout=timedelta(minutes=2))
        self.client = market_data_client
    
    def _perform_check(self) -> HealthCheckResult:
        try:
            # Check last candle timestamp
            last_update = getattr(self.client, 'last_candle_time', None)
            status, message = self._check_heartbeat(last_update)
            
            # Check latency
            latency_ms = getattr(self.client, 'last_latency_ms', 0)
            if latency_ms > 500:
                status = HealthStatus.CRITICAL
                message = f"High latency: {latency_ms}ms"
            elif latency_ms > 200:
                status = HealthStatus.WARNING
                message = f"Elevated latency: {latency_ms}ms"
            
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                details={
                    "latency_ms": latency_ms,
                    "last_update": last_update.isoformat() if last_update else None,
                },
                message=message
            )
        except Exception as e:
            return HealthCheckResult(
                module=self.module_name,
                status=HealthStatus.CRITICAL,
                details={"error": str(e)},
                message=f"Health check failed: {e}"
            )


class PolicyStoreHealthMonitor(BaseHealthMonitor):
    """Monitor for PolicyStore freshness."""
    
    def __init__(self, policy_store):
        super().__init__("policy_store", heartbeat_timeout=timedelta(minutes=10))
        self.store = policy_store
    
    def _perform_check(self) -> HealthCheckResult:
        try:
            policy = self.store.get()
            last_updated_str = policy.get("last_updated")
            
            if last_updated_str:
                last_updated = datetime.fromisoformat(last_updated_str)
                status, message = self._check_heartbeat(last_updated)
            else:
                status = HealthStatus.WARNING
                message = "No last_updated timestamp"
            
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                details={
                    "last_updated": last_updated_str,
                    "keys": list(policy.keys())
                },
                message=message
            )
        except Exception as e:
            return HealthCheckResult(
                module=self.module_name,
                status=HealthStatus.CRITICAL,
                details={"error": str(e)},
                message=f"PolicyStore check failed: {e}"
            )


class ExecutionHealthMonitor(BaseHealthMonitor):
    """Monitor for execution layer health."""
    
    def __init__(self, execution_engine):
        super().__init__("execution", heartbeat_timeout=timedelta(minutes=5))
        self.engine = execution_engine
    
    def _perform_check(self) -> HealthCheckResult:
        try:
            # Check order execution metrics
            failed_rate = getattr(self.engine, 'failed_orders_rate', 0.0)
            avg_latency = getattr(self.engine, 'avg_execution_latency_ms', 0.0)
            
            if failed_rate > 0.10:
                status = HealthStatus.CRITICAL
                message = f"High failure rate: {failed_rate:.1%}"
            elif failed_rate > 0.05:
                status = HealthStatus.WARNING
                message = f"Elevated failures: {failed_rate:.1%}"
            elif avg_latency > 1000:
                status = HealthStatus.CRITICAL
                message = f"Critical latency: {avg_latency:.0f}ms"
            elif avg_latency > 500:
                status = HealthStatus.WARNING
                message = f"High latency: {avg_latency:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Execution operating normally"
            
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                details={
                    "failed_orders_rate": failed_rate,
                    "avg_latency_ms": avg_latency,
                },
                message=message
            )
        except Exception as e:
            return HealthCheckResult(
                module=self.module_name,
                status=HealthStatus.CRITICAL,
                details={"error": str(e)},
                message=f"Execution check failed: {e}"
            )


class MSCHealthMonitor(BaseHealthMonitor):
    """Monitor for Meta Strategy Controller health."""
    
    def __init__(self, msc_scheduler):
        super().__init__("msc_ai", heartbeat_timeout=timedelta(minutes=35))
        self.scheduler = msc_scheduler
    
    def _perform_check(self) -> HealthCheckResult:
        try:
            last_run = getattr(self.scheduler, 'last_run_time', None)
            status, message = self._check_heartbeat(
                last_run,
                critical_timeout=timedelta(minutes=60)
            )
            
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                details={
                    "last_run": last_run.isoformat() if last_run else None,
                },
                message=message
            )
        except Exception as e:
            return HealthCheckResult(
                module=self.module_name,
                status=HealthStatus.CRITICAL,
                details={"error": str(e)},
                message=f"MSC check failed: {e}"
            )


class CLMHealthMonitor(BaseHealthMonitor):
    """Monitor for Continuous Learning Manager health."""
    
    def __init__(self, clm_manager):
        super().__init__("continuous_learning", heartbeat_timeout=timedelta(days=3))
        self.manager = clm_manager
    
    def _perform_check(self) -> HealthCheckResult:
        try:
            last_retrain = getattr(self.manager, 'last_retraining_time', None)
            status, message = self._check_heartbeat(
                last_retrain,
                critical_timeout=timedelta(days=7)
            )
            
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                details={
                    "last_retrain": last_retrain.isoformat() if last_retrain else None,
                },
                message=message
            )
        except Exception as e:
            return HealthCheckResult(
                module=self.module_name,
                status=HealthStatus.CRITICAL,
                details={"error": str(e)},
                message=f"CLM check failed: {e}"
            )


class OpportunityRankerHealthMonitor(BaseHealthMonitor):
    """Monitor for Opportunity Ranker health."""
    
    def __init__(self, ranker_service):
        super().__init__("opportunity_ranker", heartbeat_timeout=timedelta(minutes=6))
        self.ranker = ranker_service
    
    def _perform_check(self) -> HealthCheckResult:
        try:
            last_update = getattr(self.ranker, 'last_ranking_time', None)
            status, message = self._check_heartbeat(
                last_update,
                critical_timeout=timedelta(minutes=15)
            )
            
            # Check symbol coverage
            num_symbols = len(getattr(self.ranker, 'ranked_symbols', []))
            if num_symbols < 5:
                status = HealthStatus.WARNING
                message = f"Low symbol coverage: {num_symbols}"
            
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                details={
                    "last_update": last_update.isoformat() if last_update else None,
                    "symbols_ranked": num_symbols,
                },
                message=message
            )
        except Exception as e:
            return HealthCheckResult(
                module=self.module_name,
                status=HealthStatus.CRITICAL,
                details={"error": str(e)},
                message=f"Ranker check failed: {e}"
            )


# ============================================================================
# Integration into main.py
# ============================================================================

async def setup_health_monitoring(app_state) -> SystemHealthMonitor:
    """
    Initialize System Health Monitor with all subsystem monitors.
    
    Call this during FastAPI startup (lifespan context manager).
    
    Args:
        app_state: FastAPI app.state object with all services
        
    Returns:
        Initialized SystemHealthMonitor instance
    """
    logger.info("[SHM] Initializing System Health Monitor...")
    
    # Create monitors for all critical subsystems
    monitors = [
        MarketDataHealthMonitor(app_state.market_data_client),
        PolicyStoreHealthMonitor(app_state.policy_store),
        ExecutionHealthMonitor(app_state.execution_engine),
        MSCHealthMonitor(app_state.msc_scheduler),
        CLMHealthMonitor(app_state.clm_manager),
        OpportunityRankerHealthMonitor(app_state.opportunity_ranker),
    ]
    
    # Initialize SHM
    shm = SystemHealthMonitor(
        monitors=monitors,
        policy_store=app_state.policy_store,
        critical_threshold=1,   # 1+ critical → system CRITICAL
        warning_threshold=3,    # 3+ warnings → system WARNING
        enable_auto_write=True
    )
    
    logger.info(f"[SHM] ✅ Initialized with {len(monitors)} monitors")
    
    return shm


async def health_monitoring_loop(shm: SystemHealthMonitor, app_state):
    """
    Background task that runs health checks every minute.
    
    Responsibilities:
    - Run health checks every 60 seconds
    - React to critical failures (emergency shutdown)
    - React to warnings (adjust risk)
    - Send alerts when needed
    
    Args:
        shm: SystemHealthMonitor instance
        app_state: App state for controlling other services
    """
    logger.info("[SHM] Starting health monitoring loop...")
    
    while True:
        try:
            # Run all health checks
            summary = shm.run()
            
            # React to critical failures
            if summary.is_critical():
                logger.critical(
                    f"🚨 CRITICAL SYSTEM FAILURE: {summary.failed_modules}"
                )
                
                # Emergency shutdown
                await emergency_shutdown(app_state, summary)
                
                # Send critical alerts
                await send_critical_alert(summary)
            
            # React to warnings
            elif summary.is_degraded():
                logger.warning(
                    f"⚠️ System degraded: {summary.warning_modules}"
                )
                
                # Adjust risk to defensive mode
                await adjust_risk_for_degradation(app_state)
                
                # Send warning alerts
                await send_warning_alert(summary)
            
            # All healthy
            else:
                logger.debug("✅ System health check: All systems operational")
            
        except Exception as e:
            logger.error(f"[SHM] Health monitoring loop error: {e}")
        
        # Wait 60 seconds before next check
        await asyncio.sleep(60)


async def emergency_shutdown(app_state, summary: SystemHealthSummary):
    """
    Emergency shutdown procedure when system is CRITICAL.
    
    Steps:
    1. Stop accepting new trades
    2. Cancel open orders
    3. Close non-critical positions
    4. Disable all strategies
    5. Alert operations team
    """
    logger.critical("[SHM] 🚨 Initiating emergency shutdown...")
    
    try:
        # 1. Stop trading
        if hasattr(app_state, 'trading_enabled'):
            app_state.trading_enabled = False
            logger.critical("[SHM] ✅ Trading disabled")
        
        # 2. Cancel open orders
        if hasattr(app_state, 'execution_engine'):
            await app_state.execution_engine.cancel_all_orders()
            logger.critical("[SHM] ✅ All orders cancelled")
        
        # 3. Disable strategies
        if hasattr(app_state, 'strategy_manager'):
            await app_state.strategy_manager.disable_all_strategies()
            logger.critical("[SHM] ✅ All strategies disabled")
        
        logger.critical("[SHM] 🛑 Emergency shutdown complete")
    
    except Exception as e:
        logger.critical(f"[SHM] ❌ Emergency shutdown error: {e}")


async def adjust_risk_for_degradation(app_state):
    """Adjust risk parameters when system is degraded."""
    try:
        if hasattr(app_state, 'policy_store'):
            app_state.policy_store.patch({
                'risk_mode': 'DEFENSIVE',
                'max_positions': 3,
                'global_min_confidence': 0.75,
            })
            logger.warning("[SHM] ⚠️ Switched to DEFENSIVE mode due to degradation")
    except Exception as e:
        logger.error(f"[SHM] Failed to adjust risk: {e}")


async def send_critical_alert(summary: SystemHealthSummary):
    """Send critical alert to operations team."""
    # TODO: Implement Discord/Slack/PagerDuty integration
    logger.critical(
        f"[SHM] 🚨 CRITICAL ALERT: {len(summary.failed_modules)} modules failing"
    )


async def send_warning_alert(summary: SystemHealthSummary):
    """Send warning alert to monitoring channels."""
    # TODO: Implement Slack/Discord integration
    logger.warning(
        f"[SHM] ⚠️ WARNING: {len(summary.warning_modules)} modules degraded"
    )


# ============================================================================
# FastAPI Integration Example
# ============================================================================

"""
Add to backend/main.py:

from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.services.integrate_system_health_monitor import (
    setup_health_monitoring,
    health_monitoring_loop,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Quantum Trader backend...")
    
    # ... existing initialization ...
    
    # Initialize System Health Monitor
    shm = await setup_health_monitoring(app.state)
    app.state.system_health_monitor = shm
    
    # Start health monitoring loop in background
    health_task = asyncio.create_task(health_monitoring_loop(shm, app.state))
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    health_task.cancel()

app = FastAPI(lifespan=lifespan)

# Add health API endpoints
@app.get("/health")
async def get_health():
    shm = app.state.system_health_monitor
    summary = shm.get_last_summary()
    return summary.to_dict() if summary else {"status": "UNKNOWN"}

@app.get("/health/module/{module_name}")
async def get_module_health(module_name: str):
    shm = app.state.system_health_monitor
    status = shm.get_module_status(module_name)
    return {
        "module": module_name,
        "status": status.value if status else "UNKNOWN"
    }

@app.get("/health/history")
async def get_health_history(limit: int = 20):
    shm = app.state.system_health_monitor
    history = shm.get_history(limit=limit)
    return [h.to_dict() for h in history]
"""


# ============================================================================
# Usage from Other Services
# ============================================================================

"""
From Safety Governor:

def can_trade(self) -> bool:
    policy = self.policy_store.get()
    health = policy.get("system_health", {})
    
    # Block trading if system critical
    if health.get("status") == "CRITICAL":
        logger.warning("[SafetyGovernor] Trading blocked: system CRITICAL")
        return False
    
    return True

---

From Orchestrator:

def should_execute_signal(self, signal):
    policy = self.policy_store.get()
    health = policy.get("system_health", {})
    
    # Check if market data is healthy
    if "market_data" in health.get("failed_modules", []):
        logger.warning("[Orchestrator] Rejecting signal: market data unhealthy")
        return False
    
    return True

---

From Risk Guard:

def pre_trade_check(self, trade):
    # Query specific module status
    market_status = self.shm.get_module_status("market_data")
    
    if market_status == HealthStatus.CRITICAL:
        raise RiskViolation("Market data unavailable")
    
    return True
"""

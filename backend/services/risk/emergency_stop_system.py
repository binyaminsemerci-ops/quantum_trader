"""
Emergency Stop System (ESS) - Quantum Trader's Last Line of Defense

The Emergency Stop System provides catastrophic failure protection by:
- Monitoring critical conditions (losses, system health, execution anomalies)
- Immediately halting all trading when triggered
- Closing all positions and canceling orders
- Requiring manual reset before reactivation

This is a ZERO-TOLERANCE safety system designed to protect capital at all costs.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol, Optional

# P2-02: Import logging APIs
from backend.core import metrics_logger
from backend.core import audit_logger

# EPIC-STRESS-DASH-001: Prometheus metrics for dashboard visibility
try:
    from infra.metrics.metrics import ess_triggers_total
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# STATE & EVENTS
# ============================================================================

class ESSStatus(Enum):
    """Emergency Stop System status."""
    ACTIVE = "active"
    INACTIVE = "inactive"


class RecoveryMode(Enum):
    """[P1-05] Auto-recovery mode states."""
    EMERGENCY = "emergency"      # DD < -10%: Full stop, no trading
    PROTECTIVE = "protective"    # DD -10% to -4%: Conservative trading only
    CAUTIOUS = "cautious"        # DD -4% to -2%: Normal trading, reduced size
    NORMAL = "normal"            # DD > -2%: Full trading


@dataclass
class EmergencyState:
    """Emergency Stop state."""
    active: bool
    status: ESSStatus
    reason: Optional[str] = None
    timestamp: Optional[datetime] = None
    activation_count: int = 0
    last_check: Optional[datetime] = None
    recovery_mode: RecoveryMode = RecoveryMode.NORMAL  # [P1-05]


@dataclass
class EmergencyStopEvent:
    """Published when ESS activates."""
    reason: str = ""
    type: str = "emergency.stop"
    source: str = "ESS"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    positions_closed: int = 0
    orders_canceled: int = 0


@dataclass
class EmergencyResetEvent:
    """Published when ESS is manually reset."""
    type: str = "emergency.reset"
    reset_by: str = "manual"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EmergencyRecoveryEvent:
    """[P1-05] Published when ESS auto-recovers to a safer state."""
    type: str = "emergency.recovery"
    source: str = "ESS"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    old_mode: str = ""
    new_mode: str = ""
    current_dd_pct: float = 0.0
    recovery_type: str = "AUTO"  # AUTO or MANUAL
    trading_enabled: bool = False


# ============================================================================
# PROTOCOLS (DEPENDENCY INTERFACES)
# ============================================================================

class PolicyStore(Protocol):
    """Interface for reading/writing global policy state."""
    
    def get(self, key: str) -> dict:
        """Get policy value by key."""
        ...
    
    def set(self, key: str, value: dict) -> None:
        """Set policy value."""
        ...


class ExchangeClient(Protocol):
    """Interface for exchange operations."""
    
    async def close_all_positions(self) -> int:
        """Close all open positions. Returns count of closed positions."""
        ...
    
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of canceled orders."""
        ...
    
    async def get_account_balance(self) -> float:
        """Get current account balance."""
        ...


class EventBus(Protocol):
    """Interface for event publishing."""
    
    async def publish(self, event: object) -> None:
        """Publish an event."""
        ...


class MetricsRepository(Protocol):
    """Interface for reading metrics data."""
    
    def get_daily_pnl_percent(self) -> float:
        """Get today's PnL as percentage."""
        ...
    
    def get_equity_drawdown_percent(self) -> float:
        """Get current equity drawdown percentage."""
        ...
    
    def get_recent_sl_hits(self, minutes: int) -> int:
        """Get count of SL hits in last N minutes."""
        ...


class SystemHealthMonitor(Protocol):
    """Interface for system health status."""
    
    def get_status(self) -> str:
        """Returns: HEALTHY, DEGRADED, CRITICAL."""
        ...


class DataFeedMonitor(Protocol):
    """Interface for data feed health."""
    
    def get_last_update_time(self) -> datetime:
        """Get timestamp of last successful data update."""
        ...
    
    def is_corrupted(self) -> bool:
        """Check if recent data appears corrupted."""
        ...


# ============================================================================
# CONDITION EVALUATORS
# ============================================================================

class EmergencyConditionEvaluator(ABC):
    """Base protocol for emergency condition checkers."""
    
    @abstractmethod
    async def check(self) -> tuple[bool, Optional[str]]:
        """
        Check if emergency condition is triggered.
        
        Returns:
            (triggered: bool, reason: str | None)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this evaluator."""
        pass


class DrawdownEmergencyEvaluator(EmergencyConditionEvaluator):
    """Triggers ESS on catastrophic account losses (CRITICAL FIX #7)."""
    
    def __init__(
        self,
        metrics_repo: MetricsRepository,
        policy_store: Optional['PolicyStore'] = None,  # [CRITICAL FIX #7] Add PolicyStore
        max_daily_loss_percent: float = 10.0,  # Fallback default
        max_equity_drawdown_percent: float = 25.0,  # Fallback default
    ):
        self.metrics_repo = metrics_repo
        self.policy_store = policy_store  # [CRITICAL FIX #7]
        
        # Use fallback values if PolicyStore not provided
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_equity_drawdown_percent = max_equity_drawdown_percent
        
        if self.policy_store:
            logger.info(
                "[ESS] DrawdownEmergencyEvaluator initialized with PolicyStore "
                "(thresholds will be dynamically loaded)"
            )
        else:
            logger.warning(
                f"[ESS] DrawdownEmergencyEvaluator using HARDCODED thresholds: "
                f"max_daily_loss={max_daily_loss_percent}%, "
                f"max_equity_drawdown={max_equity_drawdown_percent}%"
            )
    
    @property
    def name(self) -> str:
        return "Drawdown Emergency"
    
    async def check(self) -> tuple[bool, Optional[str]]:
        """Check for catastrophic losses using PolicyStore thresholds (CRITICAL FIX #7)."""
        try:
            # [CRITICAL FIX #7] Load thresholds from PolicyStore
            if self.policy_store:
                try:
                    policy = await self.policy_store.get_policy()
                    active_profile = policy.get_active_profile()
                    
                    # Use PolicyStore thresholds
                    max_daily_loss = active_profile.max_drawdown_pct * 0.4  # 40% of max drawdown for daily
                    max_equity_dd = active_profile.max_drawdown_pct
                    
                    logger.debug(
                        f"[ESS] Using PolicyStore thresholds: "
                        f"max_daily_loss={max_daily_loss:.2%}, "
                        f"max_equity_drawdown={max_equity_dd:.2%}"
                    )
                except Exception as e:
                    logger.error(f"[ESS] Failed to load PolicyStore thresholds: {e}")
                    # Fall back to instance defaults
                    max_daily_loss = self.max_daily_loss_percent / 100.0
                    max_equity_dd = self.max_equity_drawdown_percent / 100.0
            else:
                # Use instance defaults (hardcoded)
                max_daily_loss = self.max_daily_loss_percent / 100.0
                max_equity_dd = self.max_equity_drawdown_percent / 100.0
            
            # Check daily loss
            daily_pnl_pct = self.metrics_repo.get_daily_pnl_percent()
            if daily_pnl_pct < -max_daily_loss * 100.0:
                return (
                    True,
                    f"Daily PnL catastrophic: {daily_pnl_pct:.2f}% "
                    f"(limit: -{max_daily_loss * 100.0:.2f}%)"
                )
            
            # Check equity drawdown
            dd_pct = self.metrics_repo.get_equity_drawdown_percent()
            if dd_pct > max_equity_dd * 100.0:
                return (
                    True,
                    f"Equity drawdown catastrophic: {dd_pct:.2f}% "
                    f"(limit: {max_equity_dd * 100.0:.2f}%)"
                )
            
            return (False, None)
        
        except Exception as e:
            logger.error(f"[ESS] DrawdownEvaluator error: {e}")
            return (False, None)


class SystemHealthEmergencyEvaluator(EmergencyConditionEvaluator):
    """Triggers ESS on critical system health failures."""
    
    def __init__(self, health_monitor: SystemHealthMonitor):
        self.health_monitor = health_monitor
    
    @property
    def name(self) -> str:
        return "System Health Emergency"
    
    async def check(self) -> tuple[bool, Optional[str]]:
        """Check system health status."""
        try:
            status = self.health_monitor.get_status()
            if status == "CRITICAL":
                return (True, f"System health status: {status}")
            return (False, None)
        except Exception as e:
            logger.error(f"[ESS] HealthEvaluator error: {e}")
            # If we can't check health, that's concerning
            return (True, f"Health monitor unavailable: {e}")


class ExecutionErrorEmergencyEvaluator(EmergencyConditionEvaluator):
    """Triggers ESS on repeated execution failures or anomalies."""
    
    def __init__(
        self,
        metrics_repo: MetricsRepository,
        max_sl_hits_per_period: int = 10,
        period_minutes: int = 15,
    ):
        self.metrics_repo = metrics_repo
        self.max_sl_hits = max_sl_hits_per_period
        self.period_minutes = period_minutes
    
    @property
    def name(self) -> str:
        return "Execution Error Emergency"
    
    async def check(self) -> tuple[bool, Optional[str]]:
        """Check for execution anomalies."""
        try:
            sl_hits = self.metrics_repo.get_recent_sl_hits(self.period_minutes)
            if sl_hits > self.max_sl_hits:
                return (
                    True,
                    f"Excessive SL hits: {sl_hits} in {self.period_minutes}min "
                    f"(limit: {self.max_sl_hits})"
                )
            return (False, None)
        except Exception as e:
            logger.error(f"[ESS] ExecutionEvaluator error: {e}")
            return (False, None)


class DataFeedEmergencyEvaluator(EmergencyConditionEvaluator):
    """Triggers ESS on data feed failures or corruption."""
    
    def __init__(
        self,
        data_feed_monitor: DataFeedMonitor,
        max_staleness_minutes: int = 5,
    ):
        self.data_feed_monitor = data_feed_monitor
        self.max_staleness = timedelta(minutes=max_staleness_minutes)
    
    @property
    def name(self) -> str:
        return "Data Feed Emergency"
    
    async def check(self) -> tuple[bool, Optional[str]]:
        """Check data feed health."""
        try:
            # Check for corrupted data
            if self.data_feed_monitor.is_corrupted():
                return (True, "Data feed corrupted")
            
            # Check for stale data
            last_update = self.data_feed_monitor.get_last_update_time()
            staleness = datetime.now(timezone.utc) - last_update
            if staleness > self.max_staleness:
                return (
                    True,
                    f"Data feed stale: {staleness.total_seconds():.0f}s "
                    f"(limit: {self.max_staleness.total_seconds():.0f}s)"
                )
            
            return (False, None)
        except Exception as e:
            logger.error(f"[ESS] DataFeedEvaluator error: {e}")
            return (False, None)  # Don't trigger on monitoring errors


class ManualTriggerEmergencyEvaluator(EmergencyConditionEvaluator):
    """Allows manual triggering of ESS."""
    
    def __init__(self):
        self._triggered = False
        self._reason: Optional[str] = None
    
    @property
    def name(self) -> str:
        return "Manual Trigger"
    
    def trigger(self, reason: str = "Manual emergency stop") -> None:
        """Manually trigger emergency stop."""
        self._triggered = True
        self._reason = reason
        logger.warning(f"[ESS] Manual trigger activated: {reason}")
    
    def reset(self) -> None:
        """Reset manual trigger."""
        self._triggered = False
        self._reason = None
    
    async def check(self) -> tuple[bool, Optional[str]]:
        """Check if manually triggered."""
        if self._triggered:
            return (True, self._reason)
        return (False, None)


# ============================================================================
# CONTROLLER
# ============================================================================

class EmergencyStopController:
    """
    Executes emergency stop procedures.
    
    Responsibilities:
    - Close all positions
    - Cancel all orders
    - Update PolicyStore
    - Publish events
    - Manage ESS state
    """
    
    def __init__(
        self,
        policy_store: PolicyStore,
        exchange: ExchangeClient,
        event_bus: EventBus,
    ):
        self.policy_store = policy_store
        self.exchange = exchange
        self.event_bus = event_bus
        self._state = self._load_state()
    
    def _load_state(self) -> EmergencyState:
        """Load ESS state from PolicyStore."""
        try:
            if not self.policy_store:
                return EmergencyState(
                    active=False,
                    status=ESSStatus.INACTIVE,
                    recovery_mode=RecoveryMode.NORMAL  # [P1-05]
                )
            
            data = self.policy_store.get("emergency_stop")
            if not data:
                return EmergencyState(
                    active=False,
                    status=ESSStatus.INACTIVE,
                    recovery_mode=RecoveryMode.NORMAL  # [P1-05]
                )
            
            # [P1-05] Load recovery mode from saved state
            recovery_mode_str = data.get("recovery_mode", "normal")
            try:
                recovery_mode = RecoveryMode(recovery_mode_str)
            except ValueError:
                recovery_mode = RecoveryMode.NORMAL
            
            return EmergencyState(
                active=data.get("active", False),
                status=ESSStatus.ACTIVE if data.get("active") else ESSStatus.INACTIVE,
                reason=data.get("reason"),
                timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
                activation_count=data.get("activation_count", 0),
                recovery_mode=recovery_mode,  # [P1-05]
            )
        except Exception as e:
            logger.error(f"[ESS] Failed to load state: {e}")
            return EmergencyState(active=False, status=ESSStatus.INACTIVE, recovery_mode=RecoveryMode.NORMAL)
    
    def _save_state(self) -> None:
        """Save ESS state to PolicyStore."""
        try:
            if not self.policy_store:
                return  # No persistence available, skip save
            
            data = {
                "active": self._state.active,
                "status": self._state.status.value,
                "reason": self._state.reason,
                "timestamp": self._state.timestamp.isoformat() if self._state.timestamp else None,
                "activation_count": self._state.activation_count,
                "recovery_mode": self._state.recovery_mode.value,  # [P1-05]
                "auto_recover": True,  # [P1-05] Enable auto-recovery
            }
            self.policy_store.set("emergency_stop", data)
            logger.info(f"[ESS] State saved: {self._state.status.value} (recovery_mode={self._state.recovery_mode.value})")
        except Exception as e:
            logger.error(f"[ESS] Failed to save state: {e}")
    
    @property
    def is_active(self) -> bool:
        """Check if ESS is currently active."""
        return self._state.active
    
    @property
    def state(self) -> EmergencyState:
        """Get current ESS state."""
        return self._state
    
    async def activate(self, reason: str) -> None:
        """
        Activate Emergency Stop System.
        
        PATCH-P0-02: Now ACTUALLY closes positions and updates PolicyStore.
        
        This will:
        1. Set ESS status to ACTIVE
        2. Update PolicyStore (emergency_mode=True, allow_new_trades=False)
        3. Close all positions with MARKET orders
        4. Cancel all open orders
        5. Publish EmergencyStopEvent
        
        Args:
            reason: Why ESS was triggered
        """
        if self._state.active:
            logger.warning(f"[ESS] Already active, ignoring activation request")
            return
        
        logger.critical(f"[ESS] ===== EMERGENCY STOP ACTIVATED ===== Reason: {reason}")
        
        # Update state
        self._state.active = True
        self._state.status = ESSStatus.ACTIVE
        self._state.reason = reason
        self._state.timestamp = datetime.utcnow()
        self._state.activation_count += 1
        self._state.recovery_mode = RecoveryMode.EMERGENCY  # [P1-05] Start recovery from EMERGENCY
        
        # EPIC-STRESS-DASH-001: Record ESS trigger metric
        if METRICS_AVAILABLE and ess_triggers_total:
            try:
                # Classify source based on reason
                source = "global_risk" if "risk" in reason.lower() else "manual"
                if "outage" in reason.lower():
                    source = "exchange_outage"
                elif "drawdown" in reason.lower():
                    source = "drawdown"
                ess_triggers_total.labels(source=source).inc()
            except Exception as e:
                logger.warning(f"[ESS] Failed to record metric: {e}")
        
        # PATCH-P0-01: Update PolicyStore FIRST (block new trades immediately)
        try:
            from backend.core.policy_store import get_policy_store
            policy_store = get_policy_store()
            await policy_store.set_emergency_mode(
                enabled=True,
                reason=reason,
                updated_by="ESS"
            )
            logger.critical("[ESS] PolicyStore emergency_mode=True, allow_new_trades=False")
        except Exception as e:
            logger.critical(f"[ESS] FAILED to update PolicyStore: {e}")
        
        positions_closed = 0
        orders_canceled = 0
        
        # PATCH-P0-02: Close all positions with MARKET orders
        try:
            logger.critical("[ESS] Closing ALL positions (MARKET orders)...")
            positions_closed = await self.exchange.close_all_positions()
            logger.critical(f"[ESS] ✓ Closed {positions_closed} positions")
        except Exception as e:
            logger.critical(f"[ESS] ✗ FAILED to close positions: {e}")
        
        # Cancel all pending orders
        try:
            logger.critical("[ESS] Canceling ALL open orders...")
            orders_canceled = await self.exchange.cancel_all_orders()
            logger.critical(f"[ESS] ✓ Canceled {orders_canceled} orders")
        except Exception as e:
            logger.critical(f"[ESS] ✗ FAILED to cancel orders: {e}")
        
        # Save state
        self._save_state()
        
        # P2-02: Log emergency trigger to audit log
        audit_logger.log_emergency_triggered(
            severity="critical",
            trigger_reason=reason,
            positions_closed=positions_closed
        )
        
        # P2-02: Record emergency stop metrics
        metrics_logger.record_counter(
            "emergency_stops_triggered",
            value=1.0,
            labels={"reason": reason[:50]}  # Truncate reason
        )
        metrics_logger.record_gauge(
            "positions_closed_emergency",
            value=float(positions_closed)
        )
        
        # Publish event
        event = EmergencyStopEvent(
            reason=reason,
            positions_closed=positions_closed,
            orders_canceled=orders_canceled,
        )
        
        try:
            await self.event_bus.publish(event)
            logger.critical(f"[ESS] Emergency event published")
        except Exception as e:
            logger.error(f"[ESS] Failed to publish event: {e}")
        
        logger.critical(
            f"[ESS] EMERGENCY STOP COMPLETE - Trading HALTED. "
            f"Manual reset required."
        )
    
    async def reset(self, reset_by: str = "manual") -> None:
        """
        Reset Emergency Stop System (manual only).
        
        PATCH-P0-01: Clears PolicyStore emergency_mode when reset.
        
        This requires human intervention and should only be done
        after the underlying issue has been resolved.
        
        Args:
            reset_by: Who/what is resetting (for audit trail)
        """
        if not self._state.active:
            logger.info("[ESS] Already inactive, nothing to reset")
            return
        
        logger.warning(f"[ESS] ===== MANUAL RESET ===== initiated by: {reset_by}")
        
        # Update state
        self._state.active = False
        self._state.status = ESSStatus.INACTIVE
        self._state.reason = None
        
        # P2-02: Log reset to audit log
        audit_logger.log_emergency_recovered(
            recovery_type="manual_reset",
            old_mode="EMERGENCY",
            new_mode="NORMAL",
            current_dd_pct=0.0  # Reset condition
        )
        
        # P2-02: Record reset metrics
        metrics_logger.record_counter(
            "emergency_resets",
            value=1.0,
            labels={"reset_by": reset_by}
        )
        
        # PATCH-P0-01: Clear PolicyStore emergency mode
        try:
            from backend.core.policy_store import get_policy_store
            policy_store = get_policy_store()
            await policy_store.set_emergency_mode(
                enabled=False,
                reason="ESS manually reset",
                updated_by=reset_by
            )
            logger.warning("[ESS] PolicyStore emergency_mode=False, allow_new_trades=True")
        except Exception as e:
            logger.error(f"[ESS] Failed to clear PolicyStore emergency mode: {e}")
        
        # Save state
        self._save_state()
        
        # Publish reset event
        event = EmergencyResetEvent(reset_by=reset_by)
        
        try:
            await self.event_bus.publish(event)
            logger.warning(f"[ESS] Reset event published")
        except Exception as e:
            logger.error(f"[ESS] Failed to publish reset event: {e}")
        
        logger.warning(f"[ESS] ESS RESET COMPLETE - Trading enabled")
    
    async def check_recovery(self, current_dd_pct: float) -> None:
        """
        [P1-05] Auto-recovery: Transition ESS state based on DD improvement.
        
        Recovery state machine:
        - DD < -10%: EMERGENCY (full stop, no trading)
        - DD -10% to -4%: PROTECTIVE (conservative trading only)
        - DD -4% to -2%: CAUTIOUS (normal trading, reduced size)
        - DD > -2%: NORMAL (full trading enabled)
        
        Args:
            current_dd_pct: Current drawdown percentage (negative = loss)
        """
        if not self._state.active and self._state.recovery_mode == RecoveryMode.NORMAL:
            # Not in recovery, nothing to do
            return
        
        old_mode = self._state.recovery_mode
        new_mode = old_mode
        
        # Determine new recovery mode based on DD
        if current_dd_pct < -10.0:
            new_mode = RecoveryMode.EMERGENCY
        elif -10.0 <= current_dd_pct < -4.0:
            new_mode = RecoveryMode.PROTECTIVE
        elif -4.0 <= current_dd_pct < -2.0:
            new_mode = RecoveryMode.CAUTIOUS
        else:  # current_dd_pct >= -2.0
            new_mode = RecoveryMode.NORMAL
        
        # No state change, skip
        if new_mode == old_mode:
            return
        
        # State transition detected
        logger.warning(
            f"[P1-05] ESS RECOVERY: {old_mode.value.upper()} → {new_mode.value.upper()} "
            f"(DD: {current_dd_pct:.2f}%)"
        )
        
        # Update state
        self._state.recovery_mode = new_mode
        
        # Update PolicyStore based on recovery mode
        try:
            from backend.core.policy_store import get_policy_store
            policy_store = get_policy_store()
            
            if new_mode == RecoveryMode.EMERGENCY:
                # Full emergency stop
                self._state.active = True
                self._state.status = ESSStatus.ACTIVE
                await policy_store.set_emergency_mode(
                    enabled=True,
                    reason=f"Auto-recovery: DD {current_dd_pct:.2f}% (EMERGENCY)",
                    updated_by="ESS-AutoRecovery"
                )
                logger.critical("[P1-05] EMERGENCY mode: Trading BLOCKED")
            
            elif new_mode == RecoveryMode.PROTECTIVE:
                # Allow conservative trades only
                self._state.active = False  # Unlock system
                self._state.status = ESSStatus.INACTIVE
                await policy_store.set_emergency_mode(
                    enabled=False,
                    reason=f"Auto-recovery: DD {current_dd_pct:.2f}% (PROTECTIVE)",
                    updated_by="ESS-AutoRecovery"
                )
                # Set protective risk params
                await policy_store.set("recovery_mode", {
                    "mode": "protective",
                    "max_position_size_usd": 100,  # Small positions only
                    "max_leverage": 5,  # Reduced leverage
                    "min_confidence": 0.75,  # High confidence only
                })
                logger.warning("[P1-05] PROTECTIVE mode: Conservative trading only")
            
            elif new_mode == RecoveryMode.CAUTIOUS:
                # Normal trading with reduced size
                self._state.active = False
                self._state.status = ESSStatus.INACTIVE
                await policy_store.set_emergency_mode(
                    enabled=False,
                    reason=f"Auto-recovery: DD {current_dd_pct:.2f}% (CAUTIOUS)",
                    updated_by="ESS-AutoRecovery"
                )
                await policy_store.set("recovery_mode", {
                    "mode": "cautious",
                    "max_position_size_usd": 300,  # Reduced positions
                    "max_leverage": 10,  # Normal leverage
                    "min_confidence": 0.55,  # Standard confidence
                })
                logger.warning("[P1-05] CAUTIOUS mode: Reduced position sizing")
            
            else:  # NORMAL
                # Full recovery - back to normal operations
                self._state.active = False
                self._state.status = ESSStatus.INACTIVE
                self._state.reason = None  # Clear emergency reason
                await policy_store.set_emergency_mode(
                    enabled=False,
                    reason="Full recovery complete",
                    updated_by="ESS-AutoRecovery"
                )
                await policy_store.set("recovery_mode", {
                    "mode": "normal",
                    "max_position_size_usd": 500,  # Full size
                    "max_leverage": 20,  # Full leverage
                    "min_confidence": 0.50,  # Normal confidence
                })
                logger.info("[P1-05] ✅ NORMAL mode: Full trading enabled")
        
        except Exception as e:
            logger.error(f"[P1-05] Failed to update PolicyStore for recovery: {e}")
        
        # Save state
        self._save_state()
        
        # Publish recovery event
        event = EmergencyRecoveryEvent(
            old_mode=old_mode.value,
            new_mode=new_mode.value,
            current_dd_pct=current_dd_pct,
            recovery_type="AUTO",
            trading_enabled=(new_mode != RecoveryMode.EMERGENCY)
        )
        
        try:
            await self.event_bus.publish(event)
            logger.warning(f"[P1-05] Recovery event published: {old_mode.value} → {new_mode.value}")
        except Exception as e:
            logger.error(f"[P1-05] Failed to publish recovery event: {e}")


# ============================================================================
# MAIN ESS RUNNER
# ============================================================================

class EmergencyStopSystem:
    """
    Main Emergency Stop System runner.
    
    Continuously evaluates emergency conditions and activates ESS when triggered.
    Runs as a background task alongside other Quantum Trader services.
    """
    
    def __init__(
        self,
        evaluators: list[EmergencyConditionEvaluator],
        controller: EmergencyStopController,
        policy_store: PolicyStore,
        check_interval_sec: int = 5,
        metrics_repo: Optional['MetricsRepository'] = None,  # [P1-05] For DD monitoring
    ):
        """
        Initialize Emergency Stop System.
        
        Args:
            evaluators: List of condition evaluators to check
            controller: Emergency stop controller
            policy_store: Policy store for state persistence
            check_interval_sec: How often to check conditions (default: 5s)
            metrics_repo: [P1-05] Metrics repository for DD monitoring
        """
        self.evaluators = evaluators
        self.controller = controller
        self.policy_store = policy_store
        self.check_interval = check_interval_sec
        self.metrics_repo = metrics_repo  # [P1-05]
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def run_forever(self) -> None:
        """
        Run ESS evaluation loop forever.
        
        Checks all emergency conditions every N seconds.
        If ESS is already active, continues monitoring but doesn't re-trigger.
        """
        self._running = True
        logger.info(
            f"[ESS] Emergency Stop System started "
            f"(check_interval={self.check_interval}s, "
            f"evaluators={len(self.evaluators)})"
        )
        
        while self._running:
            try:
                # [P1-05] Check for auto-recovery (even when ESS active)
                if self.metrics_repo:
                    try:
                        current_dd_pct = self.metrics_repo.get_equity_drawdown_percent()
                        await self.controller.check_recovery(current_dd_pct)
                    except Exception as e:
                        logger.error(f"[P1-05] Recovery check failed: {e}")
                
                # If ESS is already active, skip emergency condition evaluation
                if self.controller.is_active:
                    logger.debug("[ESS] ESS active, skipping condition checks")
                    await asyncio.sleep(self.check_interval)
                    continue
                
                # Evaluate all emergency conditions
                for evaluator in self.evaluators:
                    try:
                        triggered, reason = await evaluator.check()
                        
                        if triggered:
                            logger.critical(
                                f"[ESS] EMERGENCY CONDITION TRIGGERED: "
                                f"{evaluator.name} - {reason}"
                            )
                            await self.controller.activate(reason or evaluator.name)
                            break  # Stop checking after first trigger
                    
                    except Exception as e:
                        logger.error(
                            f"[ESS] Error in evaluator {evaluator.name}: {e}",
                            exc_info=True
                        )
                
                await asyncio.sleep(self.check_interval)
            
            except asyncio.CancelledError:
                logger.info("[ESS] Run loop cancelled")
                break
            except Exception as e:
                logger.error(f"[ESS] Error in run loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
        
        logger.info("[ESS] Emergency Stop System stopped")
    
    def start(self) -> asyncio.Task:
        """Start ESS as a background task."""
        if self._task and not self._task.done():
            logger.warning("[ESS] Already running")
            return self._task
        
        self._task = asyncio.create_task(self.run_forever())
        return self._task
    
    async def stop(self) -> None:
        """Stop ESS gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[ESS] Stopped")


# ============================================================================
# FAKE IMPLEMENTATIONS FOR TESTING
# ============================================================================

class FakePolicyStore:
    """Fake PolicyStore for testing."""
    
    def __init__(self):
        self.data: dict = {}
    
    def get(self, key: str) -> dict:
        return self.data.get(key, {})
    
    def set(self, key: str, value: dict) -> None:
        self.data[key] = value
        logger.debug(f"[FakePolicyStore] Set {key} = {value}")


class FakeExchangeClient:
    """Fake ExchangeClient for testing."""
    
    def __init__(self):
        self.positions_closed = 0
        self.orders_canceled = 0
    
    async def close_all_positions(self) -> int:
        """Simulate closing positions."""
        await asyncio.sleep(0.1)  # Simulate network delay
        self.positions_closed = 3
        logger.info("[FakeExchange] Closed 3 positions")
        return 3
    
    async def cancel_all_orders(self) -> int:
        """Simulate canceling orders."""
        await asyncio.sleep(0.1)
        self.orders_canceled = 5
        logger.info("[FakeExchange] Canceled 5 orders")
        return 5
    
    async def get_account_balance(self) -> float:
        return 10000.0


class FakeEventBus:
    """Fake EventBus for testing."""
    
    def __init__(self):
        self.events: list = []
    
    async def publish(self, event: object) -> None:
        self.events.append(event)
        logger.info(f"[FakeEventBus] Published: {event}")


class FakeMetricsRepository:
    """Fake MetricsRepository for testing."""
    
    def __init__(self):
        self.daily_pnl_pct = 0.0
        self.drawdown_pct = 0.0
        self.sl_hits = 0
    
    def get_daily_pnl_percent(self) -> float:
        return self.daily_pnl_pct
    
    def get_equity_drawdown_percent(self) -> float:
        return self.drawdown_pct
    
    def get_recent_sl_hits(self, minutes: int) -> int:
        return self.sl_hits


class FakeSystemHealthMonitor:
    """Fake SystemHealthMonitor for testing."""
    
    def __init__(self):
        self.status = "HEALTHY"
    
    def get_status(self) -> str:
        return self.status


class FakeDataFeedMonitor:
    """Fake DataFeedMonitor for testing."""
    
    def __init__(self):
        self.last_update = datetime.utcnow()
        self.corrupted = False
    
    def get_last_update_time(self) -> datetime:
        return self.last_update
    
    def is_corrupted(self) -> bool:
        return self.corrupted


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_basic_usage():
    """Basic ESS usage example."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic ESS Usage")
    print("="*80 + "\n")
    
    # Setup fake dependencies
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    # Create controller
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    # Activate ESS
    print("Activating ESS...")
    await controller.activate("Test emergency condition")
    
    # Check state
    print(f"\nESS Active: {controller.is_active}")
    print(f"Reason: {controller.state.reason}")
    print(f"Positions closed: {exchange.positions_closed}")
    print(f"Orders canceled: {exchange.orders_canceled}")
    print(f"Events published: {len(event_bus.events)}")
    
    # Reset ESS
    print("\nResetting ESS...")
    await controller.reset("admin")
    print(f"ESS Active after reset: {controller.is_active}")


async def example_drawdown_trigger():
    """Example: Drawdown triggers ESS."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Drawdown Trigger")
    print("="*80 + "\n")
    
    # Setup
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    metrics = FakeMetricsRepository()
    
    # Create evaluator and controller
    evaluator = DrawdownEmergencyEvaluator(
        metrics_repo=metrics,
        max_daily_loss_percent=10.0,
    )
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    # Simulate catastrophic loss
    print("Simulating -12% daily loss...")
    metrics.daily_pnl_pct = -12.0
    
    # Check condition
    triggered, reason = await evaluator.check()
    print(f"Triggered: {triggered}")
    print(f"Reason: {reason}")
    
    if triggered:
        await controller.activate(reason)
        print(f"\n✅ ESS activated successfully")


async def example_full_system():
    """Example: Full ESS with multiple evaluators running."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Full ESS System")
    print("="*80 + "\n")
    
    # Setup
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    metrics = FakeMetricsRepository()
    health_monitor = FakeSystemHealthMonitor()
    data_feed = FakeDataFeedMonitor()
    
    # Create evaluators
    evaluators = [
        DrawdownEmergencyEvaluator(metrics, max_daily_loss_percent=10.0),
        SystemHealthEmergencyEvaluator(health_monitor),
        ExecutionErrorEmergencyEvaluator(metrics, max_sl_hits_per_period=10),
        DataFeedEmergencyEvaluator(data_feed, max_staleness_minutes=5),
        ManualTriggerEmergencyEvaluator(),
    ]
    
    # Create controller
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    # Create ESS
    ess = EmergencyStopSystem(
        evaluators=evaluators,
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=2,
    )
    
    # Start ESS in background
    print("Starting ESS background task...")
    task = ess.start()
    
    # Run for a bit
    print("Running for 3 seconds (all conditions normal)...")
    await asyncio.sleep(3)
    
    # Trigger health failure
    print("\n🚨 Simulating system health CRITICAL...")
    health_monitor.status = "CRITICAL"
    
    # Wait for ESS to detect and activate
    await asyncio.sleep(3)
    
    # Check results
    print(f"\nESS Active: {controller.is_active}")
    print(f"Activation reason: {controller.state.reason}")
    print(f"Events published: {len(event_bus.events)}")
    
    # Stop ESS
    await ess.stop()
    print("\n✅ ESS demonstration complete")


async def example_manual_trigger():
    """Example: Manual ESS trigger."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Manual Trigger")
    print("="*80 + "\n")
    
    # Setup
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    manual_trigger = ManualTriggerEmergencyEvaluator()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    ess = EmergencyStopSystem(
        evaluators=[manual_trigger],
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=1,
    )
    
    # Start ESS
    print("Starting ESS...")
    task = ess.start()
    await asyncio.sleep(2)
    
    # Manually trigger
    print("\n🚨 Admin triggers manual emergency stop...")
    manual_trigger.trigger("Admin detected suspicious activity")
    
    # Wait for ESS to activate
    await asyncio.sleep(2)
    
    print(f"\nESS Active: {controller.is_active}")
    print(f"Reason: {controller.state.reason}")
    
    # Stop
    await ess.stop()


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def main():
    """Run all examples."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    await example_basic_usage()
    await example_drawdown_trigger()
    await example_full_system()
    await example_manual_trigger()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

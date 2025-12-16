"""
Emergency Stop System (ESS) - SPRINT 1 D3

Global safety circuit breaker that monitors risk metrics and can halt all new trading
when risk thresholds are breached. Controlled by PolicyStore.

Key Features:
- State machine: DISABLED, ARMED, TRIPPED, COOLING_DOWN
- Monitors: daily drawdown%, open loss%, execution errors
- Publishes ess.tripped events via EventBus
- Integrates with PolicyStore for dynamic thresholds
- Manual reset capability for operators

Architecture:
- Reads thresholds from PolicyStore (ess.*)
- Updates metrics via update_metrics()
- Trips when thresholds exceeded
- Blocks new orders via can_execute_orders()
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class ESSState(str, Enum):
    """Emergency Stop System states."""
    DISABLED = "DISABLED"  # ESS turned off (ess.enabled=false)
    ARMED = "ARMED"  # Normal operation, monitoring active
    TRIPPED = "TRIPPED"  # Emergency stop triggered
    COOLING_DOWN = "COOLING_DOWN"  # Waiting before auto re-arm


@dataclass
class ESSMetrics:
    """Current risk metrics tracked by ESS."""
    daily_drawdown_pct: float = 0.0  # Daily drawdown percentage (positive = loss)
    open_loss_pct: float = 0.0  # Current open positions loss percentage
    execution_errors: int = 0  # Count of recent execution errors
    last_updated: Optional[datetime] = None


class EmergencyStopSystem:
    """
    Emergency Stop System (ESS) - Global Trading Circuit Breaker
    
    Monitors risk metrics and trips when thresholds are exceeded, preventing
    new order execution until manually reset or cooldown expires.
    
    Usage:
        ess = EmergencyStopSystem(policy_store, event_bus)
        
        # Update metrics (called by risk monitoring)
        await ess.update_metrics(
            daily_drawdown_pct=2.5,
            open_loss_pct=1.8,
            execution_errors=3
        )
        
        # Check before execution
        if not await ess.can_execute_orders():
            logger.warning("ESS blocked order execution")
            return
        
        # Manual recovery
        await ess.manual_reset(user="admin")
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        clock=None
    ):
        """
        Initialize Emergency Stop System.
        
        Args:
            policy_store: PolicyStore instance for threshold configuration
            event_bus: EventBus instance for publishing events
            clock: Optional callable returning current datetime (for testing)
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        
        # State
        self.state = ESSState.ARMED
        self.trip_time: Optional[datetime] = None
        self.trip_reason: Optional[str] = None
        self.cooldown_start: Optional[datetime] = None
        
        # Metrics
        self.metrics = ESSMetrics()
        
        # Statistics
        self.trip_count = 0
        self.reset_count = 0
        
        # Check if ESS is enabled via policy
        try:
            enabled = self.policy_store.get("ess.enabled", default=True)
            if not enabled:
                self.state = ESSState.DISABLED
                logger.info("[ESS] System DISABLED via policy (ess.enabled=false)")
            else:
                logger.info("[ESS] System ARMED and monitoring")
        except Exception as e:
            logger.warning(f"[ESS] Failed to check ess.enabled policy: {e}")
            # Default to ARMED for safety
    
    async def update_metrics(
        self,
        *,
        daily_drawdown_pct: Optional[float] = None,
        open_loss_pct: Optional[float] = None,
        execution_errors: Optional[int] = None
    ) -> None:
        """
        Update risk metrics and check thresholds.
        
        Args:
            daily_drawdown_pct: Daily drawdown percentage (positive = loss)
            open_loss_pct: Current open positions loss percentage
            execution_errors: Count of recent execution errors
        """
        # Update metrics
        if daily_drawdown_pct is not None:
            self.metrics.daily_drawdown_pct = daily_drawdown_pct
        if open_loss_pct is not None:
            self.metrics.open_loss_pct = open_loss_pct
        if execution_errors is not None:
            self.metrics.execution_errors = execution_errors
        
        self.metrics.last_updated = self.clock()
        
        # If not ARMED, don't check thresholds
        if self.state != ESSState.ARMED:
            return
        
        # Get thresholds from PolicyStore
        try:
            max_daily_drawdown = self.policy_store.get("ess.max_daily_drawdown_pct", default=5.0)
            max_open_loss = self.policy_store.get("ess.max_open_loss_pct", default=10.0)
            max_errors = self.policy_store.get("ess.max_execution_errors", default=5)
        except Exception as e:
            logger.error(f"[ESS] Failed to get thresholds from PolicyStore: {e}")
            # Use safe defaults
            max_daily_drawdown = 5.0
            max_open_loss = 10.0
            max_errors = 5
        
        # Check thresholds
        if self.metrics.daily_drawdown_pct >= max_daily_drawdown:
            reason = f"Daily drawdown {self.metrics.daily_drawdown_pct:.2f}% >= {max_daily_drawdown:.2f}%"
            await self.trip(reason)
            
        elif self.metrics.open_loss_pct >= max_open_loss:
            reason = f"Open loss {self.metrics.open_loss_pct:.2f}% >= {max_open_loss:.2f}%"
            await self.trip(reason)
            
        elif self.metrics.execution_errors >= max_errors:
            reason = f"Execution errors {self.metrics.execution_errors} >= {max_errors}"
            await self.trip(reason)
    
    async def trip(self, reason: str) -> None:
        """
        Trip the emergency stop.
        
        Args:
            reason: Human-readable reason for the trip
        """
        if self.state == ESSState.TRIPPED:
            # Already tripped
            return
        
        self.state = ESSState.TRIPPED
        self.trip_time = self.clock()
        self.trip_reason = reason
        self.trip_count += 1
        
        logger.error(f"[ESS] 🚨 EMERGENCY STOP TRIPPED: {reason}")
        logger.error(f"[ESS] Metrics: {asdict(self.metrics)}")
        
        # Publish event
        try:
            await self.event_bus.publish(
                "ess.tripped",
                {
                    "reason": reason,
                    "trip_time": self.trip_time.isoformat(),
                    "metrics": asdict(self.metrics),
                    "trip_count": self.trip_count
                }
            )
        except Exception as e:
            logger.error(f"[ESS] Failed to publish ess.tripped event: {e}")
    
    async def maybe_cooldown(self) -> None:
        """
        Check if cooldown period has expired and auto re-arm if policy allows.
        
        Should be called periodically (e.g., every minute).
        """
        if self.state != ESSState.COOLING_DOWN:
            return
        
        if not self.cooldown_start:
            return
        
        # Get cooldown duration from policy
        try:
            cooldown_minutes = self.policy_store.get("ess.cooldown_minutes", default=15)
        except Exception as e:
            logger.warning(f"[ESS] Failed to get cooldown_minutes: {e}")
            cooldown_minutes = 15
        
        elapsed = (self.clock() - self.cooldown_start).total_seconds() / 60
        
        if elapsed >= cooldown_minutes:
            logger.info(f"[ESS] Cooldown expired ({elapsed:.1f}min), re-arming")
            self.state = ESSState.ARMED
            self.cooldown_start = None
            self.trip_time = None
            self.trip_reason = None
            
            # Publish re-arm event
            try:
                await self.event_bus.publish(
                    "ess.rearmed",
                    {
                        "reason": "cooldown_expired",
                        "cooldown_minutes": cooldown_minutes
                    }
                )
            except Exception as e:
                logger.error(f"[ESS] Failed to publish ess.rearmed event: {e}")
    
    async def manual_reset(self, user: str, reason: Optional[str] = None) -> bool:
        """
        Manually reset ESS to ARMED state.
        
        Args:
            user: Username/ID of operator performing reset
            reason: Optional reason for manual reset
            
        Returns:
            True if reset successful, False otherwise
        """
        if self.state not in [ESSState.TRIPPED, ESSState.COOLING_DOWN]:
            logger.warning(f"[ESS] Cannot reset from state {self.state}")
            return False
        
        # Check if manual reset is allowed by policy
        try:
            allow_manual_reset = self.policy_store.get("ess.allow_manual_reset", default=True)
        except Exception as e:
            logger.warning(f"[ESS] Failed to check ess.allow_manual_reset: {e}")
            allow_manual_reset = True
        
        if not allow_manual_reset:
            logger.error("[ESS] Manual reset not allowed by policy")
            return False
        
        prev_state = self.state
        self.state = ESSState.ARMED
        self.trip_time = None
        self.trip_reason = None
        # [SPRINT 5 - PATCH #7] Ensure cooldown timer is reset
        self.cooldown_start = None
        self.reset_count += 1
        
        logger.warning(f"[ESS] Manual reset by {user} from {prev_state} to ARMED (reset_count={self.reset_count})")
        if reason:
            logger.warning(f"[ESS] Reset reason: {reason}")
        
        # Publish event
        try:
            await self.event_bus.publish(
                "ess.manual_reset",
                {
                    "user": user,
                    "reason": reason or "manual_reset",
                    "previous_state": prev_state.value,
                    "reset_count": self.reset_count
                }
            )
        except Exception as e:
            logger.error(f"[ESS] Failed to publish ess.manual_reset event: {e}")
        
        return True
    
    async def can_execute_orders(self) -> bool:
        """
        Check if new orders can be executed.
        
        Returns:
            True if orders allowed, False if blocked
        """
        # DISABLED: always allow (ESS turned off)
        if self.state == ESSState.DISABLED:
            return True
        
        # ARMED: allow
        if self.state == ESSState.ARMED:
            return True
        
        # TRIPPED or COOLING_DOWN: block
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current ESS status.
        
        Returns:
            Dictionary with state, metrics, and statistics
        """
        return {
            "state": self.state.value,
            "can_execute": self.state in [ESSState.DISABLED, ESSState.ARMED],
            "trip_time": self.trip_time.isoformat() if self.trip_time else None,
            "trip_reason": self.trip_reason,
            "cooldown_start": self.cooldown_start.isoformat() if self.cooldown_start else None,
            "metrics": asdict(self.metrics),
            "statistics": {
                "trip_count": self.trip_count,
                "reset_count": self.reset_count
            }
        }

"""
Drawdown Circuit Breaker - Real-time monitoring (CRITICAL FIX #5)

Monitors portfolio drawdown in real-time using event-driven architecture.
Triggers circuit breaker when max drawdown threshold is breached.

Previous: Checked every 60 seconds (too slow)
New: Event-driven <1s response time
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore

logger = logging.getLogger(__name__)


class DrawdownMonitor:
    """
    Real-time drawdown monitoring with event-driven circuit breaker (CRITICAL FIX #5).
    
    Features:
    - Subscribes to position.closed and position.updated events
    - Calculates drawdown in real-time (<1s latency)
    - Triggers circuit breaker when max drawdown breached
    - Uses PolicyStore for dynamic threshold configuration
    - No polling - 100% event-driven
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        policy_store: PolicyStore,
        binance_client,
        emergency_stop_system = None,  # [P0-07] ESS instance for trade blocking
    ):
        """
        Initialize Drawdown Monitor.
        
        Args:
            event_bus: EventBus instance for event subscription
            policy_store: PolicyStore for risk thresholds
            binance_client: Binance client for position/balance queries
            emergency_stop_system: [P0-07] ESS instance for activating emergency mode
        """
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.binance_client = binance_client
        self.ess = emergency_stop_system  # [P0-07] Store ESS reference
        
        # Track peak balance for drawdown calculation
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.last_check_timestamp = datetime.now(timezone.utc)
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        
        # Subscribe to real-time events (CRITICAL FIX #5)
        self.event_bus.subscribe("position.closed", self._check_drawdown)
        self.event_bus.subscribe("position.updated", self._check_drawdown)
        self.event_bus.subscribe("balance.updated", self._check_drawdown)
        
        logger.info(
            "✅ DrawdownMonitor initialized (real-time event-driven) - "
            "subscribed to position.closed, position.updated, balance.updated"
        )
        
        # [P0-07] Log ESS integration status
        if self.ess:
            logger.info("[P0-07] ✅ ESS integration ACTIVE - will block trades on DD breach")
        else:
            logger.warning("[P0-07] ⚠️ ESS not provided - circuit breaker will only publish events")
    
    async def initialize(self) -> None:
        """Initialize peak balance from current account state."""
        try:
            account_info = await self.binance_client.futures_account()
            self.current_balance = float(account_info.get("totalWalletBalance", 0))
            self.peak_balance = self.current_balance
            
            logger.info(
                f"📊 DrawdownMonitor initialized: "
                f"current_balance=${self.current_balance:.2f}, "
                f"peak_balance=${self.peak_balance:.2f}"
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize DrawdownMonitor balance: {e}")
            # Use default values
            self.current_balance = 10000.0
            self.peak_balance = 10000.0
    
    async def _check_drawdown(self, event_data: dict) -> None:
        """
        Check drawdown in real-time on position/balance change (CRITICAL FIX #5).
        
        This method is called on EVERY position.closed, position.updated,
        and balance.updated event - ensuring <1s response time.
        
        Args:
            event_data: Event payload (symbol, pnl, balance, etc.)
        """
        try:
            # Get current balance
            account_info = await self.binance_client.futures_account()
            self.current_balance = float(account_info.get("totalWalletBalance", 0))
            
            # Update peak if new high
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
                logger.info(
                    f"📈 New peak balance: ${self.peak_balance:.2f} "
                    f"(drawdown reset to 0%)"
                )
            
            # Calculate current drawdown
            current_drawdown = self._calculate_current_drawdown()
            
            # Get max drawdown threshold from PolicyStore
            policy = await self.policy_store.get_policy()
            active_profile = policy.get_active_profile()
            max_drawdown = active_profile.max_drawdown_pct
            
            # Check if drawdown breached
            if current_drawdown > max_drawdown:
                await self._trigger_circuit_breaker(
                    current_drawdown=current_drawdown,
                    max_drawdown=max_drawdown,
                    event_data=event_data
                )
            else:
                # Log periodic status (every 10 checks)
                if not hasattr(self, '_check_count'):
                    self._check_count = 0
                self._check_count += 1
                
                if self._check_count % 10 == 0:
                    logger.debug(
                        f"📊 Drawdown status: {current_drawdown:.2%} / {max_drawdown:.2%} "
                        f"(${self.current_balance:.2f} / peak ${self.peak_balance:.2f})"
                    )
        
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
    
    def _calculate_current_drawdown(self) -> float:
        """
        Calculate current drawdown percentage.
        
        Returns:
            Drawdown as decimal (e.g., 0.15 = 15%)
        """
        if self.peak_balance == 0:
            return 0.0
        
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        return max(0.0, drawdown)  # Never negative
    
    async def _trigger_circuit_breaker(
        self,
        current_drawdown: float,
        max_drawdown: float,
        event_data: dict
    ) -> None:
        """
        Trigger circuit breaker when max drawdown breached.
        
        Args:
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum allowed drawdown
            event_data: Event that triggered the check
        """
        if self.circuit_breaker_active:
            # Already triggered, don't spam
            return
        
        self.circuit_breaker_active = True
        self.circuit_breaker_triggered_at = datetime.now(timezone.utc)
        
        logger.critical(
            f"🚨 CIRCUIT BREAKER TRIGGERED 🚨\n"
            f"   Drawdown: {current_drawdown:.2%} > Max: {max_drawdown:.2%}\n"
            f"   Current Balance: ${self.current_balance:.2f}\n"
            f"   Peak Balance: ${self.peak_balance:.2f}\n"
            f"   Loss: ${self.peak_balance - self.current_balance:.2f}\n"
            f"   Triggered by event: {event_data.get('event_type', 'unknown')}\n"
            f"   Timestamp: {self.circuit_breaker_triggered_at.isoformat()}"
        )
        
        # [P0-07] CRITICAL: Activate ESS to immediately block all trades
        if self.ess:
            try:
                await self.ess.activate(
                    reason=f"DRAWDOWN_BREACH: {current_drawdown:.1%} > {max_drawdown:.1%} "
                           f"(loss: ${self.peak_balance - self.current_balance:.2f})"
                )
                logger.critical(
                    "[P0-07] ✅ ESS ACTIVATED - All new trades BLOCKED, "
                    "existing positions closed via MARKET orders"
                )
            except Exception as e:
                logger.error(f"[P0-07] ❌ FAILED to activate ESS: {e}", exc_info=True)
        else:
            logger.warning(
                "[P0-07] ⚠️ ESS not available - circuit breaker triggered but "
                "trades NOT blocked (ESS integration missing)"
            )
        
        # Publish circuit breaker event
        await self.event_bus.publish(
            "risk.circuit_breaker.triggered",
            {
                "reason": "max_drawdown_breach",
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "loss_amount": self.peak_balance - self.current_balance,
                "triggered_at": self.circuit_breaker_triggered_at.isoformat(),
                "trigger_event": event_data,
                "ess_activated": self.ess is not None
            }
        )
    
    async def reset_circuit_breaker(self) -> None:
        """
        Reset circuit breaker (manual intervention required).
        
        Should only be called after:
        - Root cause analysis completed
        - Risk parameters adjusted
        - System validated safe to continue
        """
        if not self.circuit_breaker_active:
            logger.info("Circuit breaker not active - no reset needed")
            return
        
        logger.warning(
            f"🔄 Circuit breaker reset manually "
            f"(was triggered at {self.circuit_breaker_triggered_at.isoformat()})"
        )
        
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at = None
        
        # Publish reset event
        await self.event_bus.publish(
            "risk.circuit_breaker.reset",
            {
                "reset_at": datetime.now(timezone.utc).isoformat(),
                "manual_reset": True
            }
        )
    
    def get_status(self) -> dict:
        """
        Get current drawdown monitor status.
        
        Returns:
            Dict with current state
        """
        current_drawdown = self._calculate_current_drawdown()
        
        return {
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": current_drawdown,
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_triggered_at": (
                self.circuit_breaker_triggered_at.isoformat()
                if self.circuit_breaker_triggered_at
                else None
            ),
            "last_check": self.last_check_timestamp.isoformat(),
            "monitoring_mode": "event-driven (real-time)",
            "response_time": "<1s"
        }

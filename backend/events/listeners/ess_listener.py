"""
ESS Event Listener - SPRINT 1 D3

Subscribes to risk/portfolio/execution events and updates ESS metrics.

Events monitored:
- portfolio.pnl_update - for daily drawdown tracking
- risk.drawdown_update - for drawdown metrics
- execution.error - for execution error counting
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ESSEventListener:
    """
    Event listener for Emergency Stop System.
    
    Subscribes to risk-related events and updates ESS metrics accordingly.
    """
    
    def __init__(self, ess, event_bus):
        """
        Initialize ESS event listener.
        
        Args:
            ess: EmergencyStopSystem instance
            event_bus: EventBus instance
        """
        self.ess = ess
        self.event_bus = event_bus
        
        # Track execution errors (time-windowed)
        self.error_timestamps = []
        self.error_window_minutes = 15  # Count errors in last 15 min
        
        logger.info("[ESS Listener] Initialized")
    
    async def start(self):
        """Start listening to events."""
        try:
            # Subscribe to relevant event types
            await self.event_bus.subscribe("portfolio.pnl_update", self.handle_pnl_update)
            await self.event_bus.subscribe("risk.drawdown_update", self.handle_drawdown_update)
            await self.event_bus.subscribe("execution.error", self.handle_execution_error)
            await self.event_bus.subscribe("risk.alert", self.handle_risk_alert)
            
            logger.info("[ESS Listener] Subscribed to events")
        except Exception as e:
            logger.error(f"[ESS Listener] Failed to subscribe: {e}")
    
    async def handle_pnl_update(self, event: Dict[str, Any]):
        """
        Handle portfolio P&L update events.
        
        Expected event payload:
        {
            "daily_pnl": float,  # Daily P&L in USD
            "daily_return_pct": float,  # Daily return percentage
            "total_equity": float  # Total equity in USD
        }
        """
        try:
            payload = event.get("payload", {})
            
            # Extract daily return percentage
            daily_return_pct = payload.get("daily_return_pct", 0.0)
            
            # Convert to drawdown (positive = loss)
            daily_drawdown_pct = abs(min(0.0, daily_return_pct))
            
            # Update ESS metrics
            await self.ess.update_metrics(daily_drawdown_pct=daily_drawdown_pct)
            
        except Exception as e:
            logger.error(f"[ESS Listener] Error handling PnL update: {e}")
    
    async def handle_drawdown_update(self, event: Dict[str, Any]):
        """
        Handle drawdown update events.
        
        Expected event payload:
        {
            "daily_drawdown_pct": float,  # Daily drawdown percentage
            "open_loss_pct": float  # Current open positions loss %
        }
        """
        try:
            payload = event.get("payload", {})
            
            daily_drawdown_pct = payload.get("daily_drawdown_pct", 0.0)
            open_loss_pct = payload.get("open_loss_pct", 0.0)
            
            # Update ESS metrics
            await self.ess.update_metrics(
                daily_drawdown_pct=daily_drawdown_pct,
                open_loss_pct=open_loss_pct
            )
            
        except Exception as e:
            logger.error(f"[ESS Listener] Error handling drawdown update: {e}")
    
    async def handle_execution_error(self, event: Dict[str, Any]):
        """
        Handle execution error events.
        
        Expected event payload:
        {
            "error_type": str,
            "symbol": str,
            "timestamp": str
        }
        """
        try:
            now = datetime.utcnow()
            
            # Add error timestamp
            self.error_timestamps.append(now)
            
            # Remove old errors outside window
            cutoff = now - timedelta(minutes=self.error_window_minutes)
            self.error_timestamps = [
                ts for ts in self.error_timestamps
                if ts >= cutoff
            ]
            
            # Update ESS with error count
            error_count = len(self.error_timestamps)
            await self.ess.update_metrics(execution_errors=error_count)
            
            logger.warning(f"[ESS Listener] Execution error recorded ({error_count} in last {self.error_window_minutes}min)")
            
        except Exception as e:
            logger.error(f"[ESS Listener] Error handling execution error event: {e}")
    
    async def handle_risk_alert(self, event: Dict[str, Any]):
        """
        Handle general risk alert events.
        
        Expected event payload:
        {
            "alert_type": str,
            "severity": str,
            "metrics": dict
        }
        """
        try:
            payload = event.get("payload", {})
            metrics = payload.get("metrics", {})
            
            # Extract relevant metrics if present
            if "daily_drawdown_pct" in metrics or "open_loss_pct" in metrics:
                await self.ess.update_metrics(
                    daily_drawdown_pct=metrics.get("daily_drawdown_pct"),
                    open_loss_pct=metrics.get("open_loss_pct")
                )
            
        except Exception as e:
            logger.error(f"[ESS Listener] Error handling risk alert: {e}")

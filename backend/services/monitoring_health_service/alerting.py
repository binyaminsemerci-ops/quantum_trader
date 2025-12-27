"""
Alert Manager for Monitoring Health Service

Manages health alerts and publishes alert events to EventBus.

Author: Quantum Trader AI Team
Date: December 4, 2025
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class HealthAlert:
    """Represents a health alert."""
    
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    component: str  # Service or infra component name
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp,
            "details": self.details or {},
        }


class AlertManager:
    """
    Manages health alerts and publishes to EventBus.
    
    Responsibilities:
    - Detect alert conditions from aggregated health
    - Publish health.alert_raised events
    - Track alert history (for deduplication)
    """
    
    def __init__(
        self,
        event_bus: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.event_bus = event_bus
        self.logger = logger or logging.getLogger(__name__)
        
        # Alert tracking (simple in-memory for now)
        self._active_alerts: Dict[str, HealthAlert] = {}
        self._alert_history: List[HealthAlert] = []
    
    async def process_health(self, aggregated_health: Any) -> List[HealthAlert]:
        """
        Process aggregated health and raise alerts if needed.
        
        Args:
            aggregated_health: AggregatedHealth object
        
        Returns:
            List of new alerts raised
        """
        from .aggregators import SystemStatus
        
        new_alerts = []
        
        # Check for critical system status
        if aggregated_health.status == SystemStatus.CRITICAL:
            alert = self._create_alert(
                level=AlertLevel.CRITICAL,
                title="System Health CRITICAL",
                message=f"Critical failures detected: {', '.join(aggregated_health.critical_failures)}",
                component="system",
                details={
                    "critical_failures": aggregated_health.critical_failures,
                    "services_down": aggregated_health.services_down,
                    "infra_down": aggregated_health.infra_down,
                },
            )
            new_alerts.append(alert)
        
        # Check for degraded system status
        elif aggregated_health.status == SystemStatus.DEGRADED:
            alert = self._create_alert(
                level=AlertLevel.WARNING,
                title="System Health DEGRADED",
                message=f"System degraded: {len(aggregated_health.services_degraded)} services degraded, {len(aggregated_health.services_down)} down",
                component="system",
                details={
                    "services_degraded": aggregated_health.services_degraded,
                    "services_down": aggregated_health.services_down,
                    "infra_degraded": aggregated_health.infra_degraded,
                },
            )
            new_alerts.append(alert)
        
        # Check for specific component failures
        for service_name in aggregated_health.services_down:
            # Check if this is a critical service
            is_critical = service_name in aggregated_health.critical_failures or any(
                service_name in failure for failure in aggregated_health.critical_failures
            )
            
            alert = self._create_alert(
                level=AlertLevel.CRITICAL if is_critical else AlertLevel.WARNING,
                title=f"Service DOWN: {service_name}",
                message=f"Service {service_name} is not responding",
                component=service_name,
            )
            new_alerts.append(alert)
        
        # Check for infrastructure failures
        for infra_name in aggregated_health.infra_down:
            alert = self._create_alert(
                level=AlertLevel.CRITICAL,
                title=f"Infrastructure DOWN: {infra_name}",
                message=f"Infrastructure component {infra_name} is unavailable",
                component=infra_name,
            )
            new_alerts.append(alert)
        
        # Publish alerts
        for alert in new_alerts:
            await self._publish_alert(alert)
        
        return new_alerts
    
    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> HealthAlert:
        """Create a new health alert."""
        timestamp = datetime.now(timezone.utc).isoformat()
        alert_id = f"{component}_{level.value}_{timestamp}"
        
        alert = HealthAlert(
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            component=component,
            timestamp=timestamp,
            details=details,
        )
        
        # Track alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # Keep history bounded
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]
        
        self.logger.info(
            f"[AlertManager] Alert raised: [{level.value}] {title} - {message}"
        )
        
        return alert
    
    async def _publish_alert(self, alert: HealthAlert) -> None:
        """Publish alert to EventBus."""
        if not self.event_bus:
            self.logger.warning("[AlertManager] EventBus not available, cannot publish alert")
            return
        
        try:
            await self.event_bus.publish("health.alert_raised", alert.to_dict())
            self.logger.info(f"[AlertManager] Published alert: {alert.alert_id}")
        except Exception as e:
            self.logger.error(f"[AlertManager] Failed to publish alert: {e}")
    
    async def process_ess_tripped(self, event_data: Dict[str, Any]) -> None:
        """
        Handle ess.tripped event.
        
        Args:
            event_data: Event payload from ESS
        """
        reason = event_data.get("reason", "Unknown")
        severity = event_data.get("severity", "CRITICAL")
        
        alert = self._create_alert(
            level=AlertLevel.CRITICAL,
            title="Emergency Stop System TRIPPED",
            message=f"ESS activated: {reason}",
            component="emergency_stop_system",
            details=event_data,
        )
        
        await self._publish_alert(alert)
        
        self.logger.critical(f"[AlertManager] ESS TRIPPED: {reason}")
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[HealthAlert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]
    
    def clear_alert(self, alert_id: str) -> bool:
        """Clear/acknowledge an active alert."""
        if alert_id in self._active_alerts:
            del self._active_alerts[alert_id]
            self.logger.info(f"[AlertManager] Alert cleared: {alert_id}")
            return True
        return False

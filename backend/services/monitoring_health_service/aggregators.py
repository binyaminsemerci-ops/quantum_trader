"""
Health Aggregators for Monitoring Health Service

Aggregates health snapshots from collectors and determines overall system status.

Author: Quantum Trader AI Team
Date: December 4, 2025
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemStatus(str, Enum):
    """Overall system health status."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class AggregatedHealth:
    """Aggregated health summary with global status."""
    
    status: SystemStatus
    timestamp: str
    
    # Breakdown by status
    services_ok: List[str] = field(default_factory=list)
    services_degraded: List[str] = field(default_factory=list)
    services_down: List[str] = field(default_factory=list)
    
    infra_ok: List[str] = field(default_factory=list)
    infra_degraded: List[str] = field(default_factory=list)
    infra_down: List[str] = field(default_factory=list)
    
    # Critical failures
    critical_failures: List[str] = field(default_factory=list)
    
    # Performance metrics
    avg_service_latency_ms: Optional[float] = None
    max_service_latency_ms: Optional[float] = None
    
    # Detailed snapshot
    snapshot: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "services": {
                "ok": self.services_ok,
                "degraded": self.services_degraded,
                "down": self.services_down,
            },
            "infrastructure": {
                "ok": self.infra_ok,
                "degraded": self.infra_degraded,
                "down": self.infra_down,
            },
            "critical_failures": self.critical_failures,
            "performance": {
                "avg_service_latency_ms": self.avg_service_latency_ms,
                "max_service_latency_ms": self.max_service_latency_ms,
            },
            "snapshot": self.snapshot,
        }


class HealthAggregator:
    """
    Aggregates health snapshots and determines global system status.
    
    Status determination logic:
    - CRITICAL: Any critical service is DOWN OR any infra component is DOWN
    - DEGRADED: Any service is DEGRADED/DOWN (non-critical) OR high latency
    - OK: All services and infra are OK
    - UNKNOWN: No data or inconsistent state
    """
    
    LATENCY_WARNING_THRESHOLD_MS = 1000.0  # 1 second
    LATENCY_CRITICAL_THRESHOLD_MS = 5000.0  # 5 seconds
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def aggregate(self, snapshot: Dict[str, Any]) -> AggregatedHealth:
        """
        Aggregate a health snapshot into global status.
        
        Args:
            snapshot: Raw snapshot from HealthCollector
        
        Returns:
            AggregatedHealth with computed global status
        """
        self.logger.info("[HealthAggregator] Aggregating health snapshot")
        
        timestamp = snapshot.get("timestamp", datetime.now(timezone.utc).isoformat())
        services = snapshot.get("services", {})
        infra = snapshot.get("infra", {})
        
        # Categorize services
        services_ok = []
        services_degraded = []
        services_down = []
        critical_failures = []
        
        latencies = []
        
        for name, health in services.items():
            status = health.get("status", "UNKNOWN")
            is_critical = health.get("critical", True)
            latency = health.get("latency_ms")
            
            if latency is not None:
                latencies.append(latency)
            
            if status == "OK":
                services_ok.append(name)
            elif status == "DEGRADED":
                services_degraded.append(name)
            elif status == "DOWN":
                services_down.append(name)
                if is_critical:
                    critical_failures.append(f"Service: {name}")
        
        # Categorize infrastructure
        infra_ok = []
        infra_degraded = []
        infra_down = []
        
        for name, health in infra.items():
            status = health.get("status", "UNKNOWN")
            
            if status == "OK":
                infra_ok.append(name)
            elif status == "DEGRADED":
                infra_degraded.append(name)
            elif status == "DOWN":
                infra_down.append(name)
                critical_failures.append(f"Infrastructure: {name}")
        
        # Compute performance metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None
        
        # Determine global status
        global_status = self._determine_global_status(
            critical_failures=critical_failures,
            services_down=services_down,
            services_degraded=services_degraded,
            infra_down=infra_down,
            infra_degraded=infra_degraded,
            max_latency=max_latency,
        )
        
        self.logger.info(
            f"[HealthAggregator] Global status: {global_status.value} "
            f"(Critical failures: {len(critical_failures)})"
        )
        
        return AggregatedHealth(
            status=global_status,
            timestamp=timestamp,
            services_ok=services_ok,
            services_degraded=services_degraded,
            services_down=services_down,
            infra_ok=infra_ok,
            infra_degraded=infra_degraded,
            infra_down=infra_down,
            critical_failures=critical_failures,
            avg_service_latency_ms=round(avg_latency, 2) if avg_latency else None,
            max_service_latency_ms=round(max_latency, 2) if max_latency else None,
            snapshot=snapshot,
        )
    
    def _determine_global_status(
        self,
        critical_failures: List[str],
        services_down: List[str],
        services_degraded: List[str],
        infra_down: List[str],
        infra_degraded: List[str],
        max_latency: Optional[float],
    ) -> SystemStatus:
        """Determine global system status based on component states."""
        
        # CRITICAL: Any critical failures
        if critical_failures:
            return SystemStatus.CRITICAL
        
        # CRITICAL: Any infrastructure down
        if infra_down:
            return SystemStatus.CRITICAL
        
        # CRITICAL: Extreme latency
        if max_latency and max_latency > self.LATENCY_CRITICAL_THRESHOLD_MS:
            return SystemStatus.CRITICAL
        
        # DEGRADED: Any service down (non-critical) or degraded
        if services_down or services_degraded:
            return SystemStatus.DEGRADED
        
        # DEGRADED: Any infrastructure degraded
        if infra_degraded:
            return SystemStatus.DEGRADED
        
        # DEGRADED: High latency
        if max_latency and max_latency > self.LATENCY_WARNING_THRESHOLD_MS:
            return SystemStatus.DEGRADED
        
        # OK: Everything healthy
        return SystemStatus.OK
    
    def compute_trend(self, snapshots: List[AggregatedHealth]) -> Dict[str, Any]:
        """
        Compute trends from multiple snapshots.
        
        Args:
            snapshots: List of recent AggregatedHealth objects
        
        Returns:
            Trend analysis (error rate, status changes, etc.)
        """
        if not snapshots:
            return {"trend": "unknown", "reason": "No data"}
        
        # Count status changes
        status_counts = {
            SystemStatus.OK: 0,
            SystemStatus.DEGRADED: 0,
            SystemStatus.CRITICAL: 0,
        }
        
        for snapshot in snapshots:
            status_counts[snapshot.status] = status_counts.get(snapshot.status, 0) + 1
        
        # Determine trend
        recent_count = min(3, len(snapshots))
        recent_statuses = [s.status for s in snapshots[-recent_count:]]
        
        if all(s == SystemStatus.CRITICAL for s in recent_statuses):
            trend = "degrading"
            reason = "Sustained CRITICAL status"
        elif all(s == SystemStatus.OK for s in recent_statuses):
            trend = "healthy"
            reason = "Sustained OK status"
        elif recent_statuses[-1] == SystemStatus.OK and recent_statuses[0] != SystemStatus.OK:
            trend = "recovering"
            reason = "Status improved to OK"
        elif recent_statuses[-1] == SystemStatus.CRITICAL and recent_statuses[0] != SystemStatus.CRITICAL:
            trend = "degrading"
            reason = "Status degraded to CRITICAL"
        else:
            trend = "stable"
            reason = "Status unchanged"
        
        return {
            "trend": trend,
            "reason": reason,
            "status_distribution": {
                status.value: count
                for status, count in status_counts.items()
            },
            "sample_size": len(snapshots),
        }

"""
Standardized Health Check Contract
SPRINT 3 - Part 2: Infrastructure Hardening

This module defines the standard health check format used across all microservices.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class DependencyStatus(str, Enum):
    """Dependency health status."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    NOT_APPLICABLE = "N/A"


@dataclass
class DependencyHealth:
    """Health status for a single dependency."""
    status: DependencyStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result = {"status": self.status.value}
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.error:
            result["error"] = self.error
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class ServiceHealth:
    """
    Standardized health check response for all microservices.
    
    Contract:
    {
        "service": "execution-service",
        "status": "OK" | "DEGRADED" | "DOWN",
        "version": "1.0.0",
        "timestamp": "2025-12-04T10:30:00Z",
        "uptime_seconds": 3600,
        "dependencies": {
            "redis": {
                "status": "OK",
                "latency_ms": 1.5
            },
            "postgres": {
                "status": "DEGRADED",
                "latency_ms": 250.3,
                "error": "Slow query"
            },
            "binance_api": {
                "status": "OK",
                "latency_ms": 45.2
            },
            "eventbus": {
                "status": "OK"
            }
        },
        "metrics": {
            "active_connections": 5,
            "total_requests": 1000
        }
    }
    """
    service: str
    status: HealthStatus
    version: str
    timestamp: str
    uptime_seconds: float
    dependencies: Dict[str, DependencyHealth]
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "service": self.service,
            "status": self.status.value,
            "version": self.version,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "dependencies": {
                name: dep.to_dict() 
                for name, dep in self.dependencies.items()
            }
        }
        
        if self.metrics:
            result["metrics"] = self.metrics
        
        if self.error:
            result["error"] = self.error
        
        return result
    
    @classmethod
    def create(
        cls,
        service_name: str,
        version: str,
        start_time: datetime,
        dependencies: Dict[str, DependencyHealth],
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> "ServiceHealth":
        """
        Create a ServiceHealth instance with automatic status calculation.
        
        Status Logic:
        - OK: All dependencies OK
        - DEGRADED: Some dependencies DEGRADED or DOWN (but service still functional)
        - DOWN: Critical dependencies DOWN (service non-functional)
        """
        # Calculate overall status
        dep_statuses = [dep.status for dep in dependencies.values()]
        
        if DependencyStatus.DOWN in dep_statuses:
            # Check if it's a critical dependency (redis, eventbus)
            critical_down = any(
                name in ["redis", "eventbus", "postgres"] and dep.status == DependencyStatus.DOWN
                for name, dep in dependencies.items()
            )
            overall_status = HealthStatus.DOWN if critical_down else HealthStatus.DEGRADED
        elif DependencyStatus.DEGRADED in dep_statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.OK
        
        # Override if there's an explicit error
        if error:
            overall_status = HealthStatus.DOWN
        
        now = datetime.now(timezone.utc)
        uptime = (now - start_time).total_seconds()
        
        return cls(
            service=service_name,
            status=overall_status,
            version=version,
            timestamp=now.isoformat(),
            uptime_seconds=uptime,
            dependencies=dependencies,
            metrics=metrics,
            error=error
        )


async def check_redis_health(redis_client, timeout: float = 2.0) -> DependencyHealth:
    """
    Check Redis health with latency measurement.
    
    Args:
        redis_client: Redis client instance
        timeout: Timeout in seconds
    
    Returns:
        DependencyHealth with status and latency
    """
    import time
    
    try:
        start = time.time()
        await asyncio.wait_for(redis_client.ping(), timeout=timeout)
        latency_ms = (time.time() - start) * 1000
        
        # Degraded if slow
        if latency_ms > 100:
            return DependencyHealth(
                status=DependencyStatus.DEGRADED,
                latency_ms=latency_ms,
                details={"warning": "High latency"}
            )
        
        return DependencyHealth(
            status=DependencyStatus.OK,
            latency_ms=latency_ms
        )
    
    except asyncio.TimeoutError:
        return DependencyHealth(
            status=DependencyStatus.DOWN,
            error=f"Timeout after {timeout}s"
        )
    
    except Exception as e:
        return DependencyHealth(
            status=DependencyStatus.DOWN,
            error=str(e)
        )


async def check_postgres_health(connection_pool, timeout: float = 2.0) -> DependencyHealth:
    """
    Check Postgres health with latency measurement.
    
    Args:
        connection_pool: Database connection pool
        timeout: Timeout in seconds
    
    Returns:
        DependencyHealth with status and latency
    """
    import time
    
    try:
        start = time.time()
        
        # Simple query
        async with asyncio.timeout(timeout):
            # Get connection and execute simple query
            # Implementation depends on your DB library
            # Example: await connection_pool.execute("SELECT 1")
            pass  # Placeholder - implement based on your DB library
        
        latency_ms = (time.time() - start) * 1000
        
        if latency_ms > 200:
            return DependencyHealth(
                status=DependencyStatus.DEGRADED,
                latency_ms=latency_ms,
                details={"warning": "Slow query"}
            )
        
        return DependencyHealth(
            status=DependencyStatus.OK,
            latency_ms=latency_ms
        )
    
    except asyncio.TimeoutError:
        return DependencyHealth(
            status=DependencyStatus.DOWN,
            error=f"Timeout after {timeout}s"
        )
    
    except Exception as e:
        return DependencyHealth(
            status=DependencyStatus.DOWN,
            error=str(e)
        )


async def check_http_endpoint_health(
    http_client,
    url: str,
    timeout: float = 5.0
) -> DependencyHealth:
    """
    Check HTTP endpoint health with latency measurement.
    
    Args:
        http_client: HTTP client (httpx.AsyncClient)
        url: URL to check
        timeout: Timeout in seconds
    
    Returns:
        DependencyHealth with status and latency
    """
    import time
    
    try:
        start = time.time()
        response = await http_client.get(url, timeout=timeout)
        latency_ms = (time.time() - start) * 1000
        
        if response.status_code != 200:
            return DependencyHealth(
                status=DependencyStatus.DEGRADED,
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}"
            )
        
        if latency_ms > 500:
            return DependencyHealth(
                status=DependencyStatus.DEGRADED,
                latency_ms=latency_ms,
                details={"warning": "High latency"}
            )
        
        return DependencyHealth(
            status=DependencyStatus.OK,
            latency_ms=latency_ms
        )
    
    except asyncio.TimeoutError:
        return DependencyHealth(
            status=DependencyStatus.DOWN,
            error=f"Timeout after {timeout}s"
        )
    
    except Exception as e:
        return DependencyHealth(
            status=DependencyStatus.DOWN,
            error=str(e)[:100]  # Truncate long errors
        )


# Make asyncio available for helper functions
import asyncio

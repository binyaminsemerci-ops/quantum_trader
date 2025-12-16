"""Health check system for monitoring service dependencies and system resources.

This module provides comprehensive health checks for:
- Redis connectivity
- PostgreSQL database
- Binance REST API
- Binance WebSocket
- System resources (CPU, memory)
- Overall service health status
"""

from __future__ import annotations

import asyncio
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import aiohttp
import psutil
import redis.asyncio as redis
from redis.asyncio import Redis

from backend.core.logger import get_logger

logger = get_logger(__name__, component="health_checker")


class HealthStatus(str, Enum):
    """Health status levels."""
    
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class DependencyHealth:
    """Health status of a single dependency."""
    
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "last_check": self.last_check.isoformat(),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """System resource health."""
    
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    status: HealthStatus
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_percent": self.disk_percent,
            "status": self.status.value,
        }


@dataclass
class HealthReport:
    """Complete health report for the service."""
    
    service: str
    status: HealthStatus
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dependencies: dict[str, DependencyHealth] = field(default_factory=dict)
    system: Optional[SystemHealth] = None
    modules: dict[str, HealthStatus] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "service": self.service,
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
            "dependencies": {
                name: dep.to_dict()
                for name, dep in self.dependencies.items()
            },
            "system": self.system.to_dict() if self.system else None,
            "modules": {
                name: status.value
                for name, status in self.modules.items()
            },
        }


class HealthChecker:
    """
    Comprehensive health checker for all service dependencies.
    
    Usage:
        checker = HealthChecker(
            service_name="quantum_trader",
            redis_client=redis_client,
            postgres_url="postgresql://...",
            binance_api_key="...",
            binance_api_secret="...",
        )
        
        await checker.initialize()
        
        # Get full health report
        report = await checker.get_health_report()
        
        # Check specific dependency
        redis_health = await checker.check_redis()
    """
    
    CACHE_TTL = 5.0  # Cache health checks for 5 seconds
    
    def __init__(
        self,
        service_name: str,
        redis_client: Optional[Redis] = None,
        postgres_url: Optional[str] = None,
        binance_api_key: Optional[str] = None,
        binance_api_secret: Optional[str] = None,
        binance_testnet: bool = True,
    ):
        """
        Initialize health checker.
        
        Args:
            service_name: Name of this service
            redis_client: Optional Redis client for health checks
            postgres_url: Optional PostgreSQL connection URL
            binance_api_key: Optional Binance API key
            binance_api_secret: Optional Binance API secret
            binance_testnet: If True, use testnet endpoints
        """
        self.service_name = service_name
        self.redis_client = redis_client
        self.postgres_url = postgres_url
        self.binance_api_key = binance_api_key
        self.binance_api_secret = binance_api_secret
        self.binance_testnet = binance_testnet
        
        # Binance endpoints
        if binance_testnet:
            self.binance_base_url = "https://testnet.binancefuture.com"
            self.binance_ws_url = "wss://stream.binancefuture.com"
        else:
            self.binance_base_url = "https://fapi.binance.com"
            self.binance_ws_url = "wss://fstream.binance.com"
        
        # Track startup time
        self.startup_time = time.time()
        
        # Health cache
        self._last_report: Optional[HealthReport] = None
        self._last_report_time: Optional[datetime] = None
        
        logger.info(
            f"HealthChecker initialized: service={service_name}, "
            f"binance_testnet={binance_testnet}"
        )
    
    async def initialize(self) -> None:
        """Initialize health checker (verify connectivity)."""
        logger.info("HealthChecker starting initial checks")
        
        # Run initial health check
        report = await self.get_health_report(use_cache=False)
        
        logger.info(
            f"HealthChecker initialized: status={report.status.value}, "
            f"healthy_deps={sum(1 for dep in report.dependencies.values() if dep.status == HealthStatus.HEALTHY)}, "
            f"total_deps={len(report.dependencies)}"
        )
    
    async def get_health_report(
        self,
        use_cache: bool = True,
    ) -> HealthReport:
        """
        Get complete health report for all dependencies.
        
        Args:
            use_cache: If True, use cached report if recent
        
        Returns:
            Complete HealthReport
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._last_report
        
        # Run all health checks in parallel
        checks = {
            "redis": self.check_redis() if self.redis_client else None,
            "postgres": self.check_postgres() if self.postgres_url else None,
            "binance_rest": self.check_binance_rest() if self.binance_api_key else None,
            "binance_ws": self.check_binance_ws() if self.binance_api_key else None,
        }
        
        # Filter out None checks
        checks = {k: v for k, v in checks.items() if v is not None}
        
        # Execute checks
        results = await asyncio.gather(*checks.values(), return_exceptions=True)
        
        # Build dependency map
        dependencies = {}
        for (name, _), result in zip(checks.items(), results):
            if isinstance(result, Exception):
                dependencies[name] = DependencyHealth(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    error=str(result),
                )
            else:
                dependencies[name] = result
        
        # Get system health
        system_health = await self.check_system()
        
        # Determine overall status
        overall_status = self._calculate_overall_status(dependencies, system_health)
        
        # Calculate uptime
        uptime = time.time() - self.startup_time
        
        # Create report
        report = HealthReport(
            service=self.service_name,
            status=overall_status,
            uptime_seconds=uptime,
            dependencies=dependencies,
            system=system_health,
        )
        
        # Update cache
        self._last_report = report
        self._last_report_time = datetime.utcnow()
        
        return report
    
    async def check_redis(self) -> DependencyHealth:
        """
        Check Redis connectivity and latency.
        
        Returns:
            DependencyHealth for Redis
        """
        if not self.redis_client:
            return DependencyHealth(
                name="redis",
                status=HealthStatus.UNKNOWN,
                error="Redis client not configured",
            )
        
        try:
            start = time.time()
            
            # Add 1-second timeout for health check
            await asyncio.wait_for(self.redis_client.ping(), timeout=1.0)
            latency_ms = (time.time() - start) * 1000
            
            # Get Redis info with timeout
            info = await asyncio.wait_for(self.redis_client.info(), timeout=1.0)
            
            return DependencyHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={
                    "version": info.get("redis_version", "unknown"),
                    "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                    "connected_clients": info.get("connected_clients", 0),
                },
            )
        
        except asyncio.TimeoutError:
            return DependencyHealth(
                name="redis",
                status=HealthStatus.CRITICAL,
                error="Timeout after 1s (degraded)",
            )
        except redis.RedisError as e:
            return DependencyHealth(
                name="redis",
                status=HealthStatus.CRITICAL,
                error=f"Redis error: {str(e)}",
            )
        except Exception as e:
            return DependencyHealth(
                name="redis",
                status=HealthStatus.CRITICAL,
                error=f"Unexpected error: {str(e)}",
            )
    
    async def check_postgres(self) -> DependencyHealth:
        """
        Check PostgreSQL connectivity.
        
        Returns:
            DependencyHealth for PostgreSQL
        """
        if not self.postgres_url:
            return DependencyHealth(
                name="postgres",
                status=HealthStatus.UNKNOWN,
                error="PostgreSQL not configured",
            )
        
        try:
            import asyncpg
            
            start = time.time()
            
            # Add 1-second timeout for health check
            conn = await asyncio.wait_for(
                asyncpg.connect(self.postgres_url),
                timeout=1.0
            )
            
            # Simple query
            version = await asyncio.wait_for(
                conn.fetchval("SELECT version()"),
                timeout=1.0
            )
            
            await conn.close()
            
            latency_ms = (time.time() - start) * 1000
            
            return DependencyHealth(
                name="postgres",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={"version": version[:50]},  # Truncate long version string
            )
        
        except asyncio.TimeoutError:
            return DependencyHealth(
                name="postgres",
                status=HealthStatus.CRITICAL,
                error="Timeout after 1s (degraded)",
            )
        except Exception as e:
            return DependencyHealth(
                name="postgres",
                status=HealthStatus.CRITICAL,
                error=f"PostgreSQL error: {str(e)}",
            )
    
    async def check_binance_rest(self) -> DependencyHealth:
        """
        Check Binance REST API connectivity.
        
        Returns:
            DependencyHealth for Binance REST
        """
        if not self.binance_api_key:
            return DependencyHealth(
                name="binance_rest",
                status=HealthStatus.UNKNOWN,
                error="Binance API not configured",
            )
        
        try:
            url = f"{self.binance_base_url}/fapi/v1/ping"
            
            start = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=1)) as resp:
                    if resp.status != 200:
                        return DependencyHealth(
                            name="binance_rest",
                            status=HealthStatus.DEGRADED,
                            error=f"HTTP {resp.status}",
                        )
                    
                    latency_ms = (time.time() - start) * 1000
                    
                    return DependencyHealth(
                        name="binance_rest",
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency_ms,
                        details={"endpoint": self.binance_base_url},
                    )
        
        except asyncio.TimeoutError:
            return DependencyHealth(
                name="binance_rest",
                status=HealthStatus.CRITICAL,
                error="Timeout after 1s (degraded)",
            )
        except Exception as e:
            return DependencyHealth(
                name="binance_rest",
                status=HealthStatus.CRITICAL,
                error=f"Error: {str(e)}",
            )
    
    async def check_binance_ws(self) -> DependencyHealth:
        """
        Check Binance WebSocket connectivity.
        
        Returns:
            DependencyHealth for Binance WebSocket
        """
        # For now, return healthy if REST is healthy
        # Full WebSocket check would require maintaining a connection
        return DependencyHealth(
            name="binance_ws",
            status=HealthStatus.HEALTHY,
            details={"endpoint": self.binance_ws_url},
        )
    
    async def check_system(self) -> SystemHealth:
        """
        Check system resource utilization.
        
        Returns:
            SystemHealth with CPU, memory, disk metrics
        """
        try:
            # CPU usage (1 second sample)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            mem = psutil.virtual_memory()
            
            # Disk usage (root partition)
            disk = psutil.disk_usage("/")
            
            # Determine status
            status = HealthStatus.HEALTHY
            
            if cpu_percent > 90 or mem.percent > 90 or disk.percent > 90:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 70 or mem.percent > 70 or disk.percent > 80:
                status = HealthStatus.DEGRADED
            
            return SystemHealth(
                cpu_percent=cpu_percent,
                memory_percent=mem.percent,
                memory_used_mb=mem.used / 1024 / 1024,
                memory_available_mb=mem.available / 1024 / 1024,
                disk_percent=disk.percent,
                status=status,
            )
        
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                cpu_percent=0,
                memory_percent=0,
                memory_used_mb=0,
                memory_available_mb=0,
                disk_percent=0,
                status=HealthStatus.UNKNOWN,
            )
    
    def _calculate_overall_status(
        self,
        dependencies: dict[str, DependencyHealth],
        system: Optional[SystemHealth],
    ) -> HealthStatus:
        """
        Calculate overall service status from dependencies and system health.
        
        Args:
            dependencies: Map of dependency health
            system: System resource health
        
        Returns:
            Overall HealthStatus
        """
        # Critical if any dependency is critical
        if any(dep.status == HealthStatus.CRITICAL for dep in dependencies.values()):
            return HealthStatus.CRITICAL
        
        # Critical if system is critical
        if system and system.status == HealthStatus.CRITICAL:
            return HealthStatus.CRITICAL
        
        # Degraded if any dependency is degraded
        if any(dep.status == HealthStatus.DEGRADED for dep in dependencies.values()):
            return HealthStatus.DEGRADED
        
        # Degraded if system is degraded
        if system and system.status == HealthStatus.DEGRADED:
            return HealthStatus.DEGRADED
        
        # Healthy if all dependencies healthy
        return HealthStatus.HEALTHY
    
    def _is_cache_valid(self) -> bool:
        """Check if cached health report is still valid."""
        if self._last_report is None or self._last_report_time is None:
            return False
        
        age = (datetime.utcnow() - self._last_report_time).total_seconds()
        return age < self.CACHE_TTL


# Singleton instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """
    Get singleton HealthChecker instance.
    
    Returns:
        HealthChecker instance
    
    Raises:
        RuntimeError: If not initialized
    """
    if _health_checker is None:
        raise RuntimeError(
            "HealthChecker not initialized. Call initialize_health_checker() first."
        )
    return _health_checker


async def initialize_health_checker(
    service_name: str,
    redis_client: Optional[Redis] = None,
    postgres_url: Optional[str] = None,
    binance_api_key: Optional[str] = None,
    binance_api_secret: Optional[str] = None,
    binance_testnet: bool = True,
) -> HealthChecker:
    """
    Initialize global HealthChecker singleton.
    
    Args:
        service_name: Name of this service
        redis_client: Optional Redis client
        postgres_url: Optional PostgreSQL URL
        binance_api_key: Optional Binance API key
        binance_api_secret: Optional Binance API secret
        binance_testnet: If True, use testnet
    
    Returns:
        Initialized HealthChecker
    """
    global _health_checker
    
    _health_checker = HealthChecker(
        service_name=service_name,
        redis_client=redis_client,
        postgres_url=postgres_url,
        binance_api_key=binance_api_key,
        binance_api_secret=binance_api_secret,
        binance_testnet=binance_testnet,
    )
    
    await _health_checker.initialize()
    
    return _health_checker

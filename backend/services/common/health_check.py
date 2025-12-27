"""
Health Check Infrastructure for AI Modules
Standardized health check endpoints and monitoring
"""
import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    module_name: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Any]
    errors: list
    warnings: list
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class HealthChecker:
    """
    Base health checker for AI modules.
    Provides standardized health check functionality.
    """
    
    def __init__(self, module_name: str):
        """
        Initialize health checker
        
        Args:
            module_name: Name of the module to monitor
        """
        self.module_name = module_name
        self.start_time = time.time()
        self.last_error: Optional[str] = None
        self.error_count = 0
        self.warning_count = 0
        
        logger.info(f"Health checker initialized for {module_name}")
    
    def check_health(self) -> HealthCheckResult:
        """
        Perform health check
        
        Returns:
            HealthCheckResult with current status
        """
        errors = []
        warnings = []
        checks = {}
        
        # Basic checks
        checks["uptime"] = self._check_uptime()
        checks["memory"] = self._check_memory()
        checks["cpu"] = self._check_cpu()
        
        # Module-specific checks (override in subclass)
        module_checks = self._module_specific_checks()
        checks.update(module_checks)
        
        # Determine overall status
        if errors:
            status = HealthStatus.UNHEALTHY
        elif warnings:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return HealthCheckResult(
            status=status,
            module_name=self.module_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=time.time() - self.start_time,
            checks=checks,
            errors=errors,
            warnings=warnings
        )
    
    def _check_uptime(self) -> Dict[str, Any]:
        """Check module uptime"""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "status": "ok"
        }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            return {
                "memory_mb": round(memory_mb, 2),
                "status": "ok" if memory_mb < 500 else "warning"
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            return {
                "cpu_percent": round(cpu_percent, 2),
                "status": "ok" if cpu_percent < 50 else "warning"
            }
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _module_specific_checks(self) -> Dict[str, Any]:
        """
        Module-specific health checks.
        Override in subclass.
        
        Returns:
            Dictionary of check results
        """
        return {}
    
    def record_error(self, error: str):
        """Record an error"""
        self.last_error = error
        self.error_count += 1
        logger.error(f"[{self.module_name}] Error: {error}")
    
    def record_warning(self, warning: str):
        """Record a warning"""
        self.warning_count += 1
        logger.warning(f"[{self.module_name}] Warning: {warning}")
    
    def reset_counters(self):
        """Reset error and warning counters"""
        self.error_count = 0
        self.warning_count = 0
        logger.info(f"[{self.module_name}] Counters reset")


def create_health_endpoint(health_checker: HealthChecker):
    """
    Create a FastAPI health check endpoint
    
    Args:
        health_checker: HealthChecker instance
        
    Returns:
        FastAPI endpoint function
    """
    async def health():
        """Health check endpoint"""
        result = health_checker.check_health()
        
        # Return appropriate HTTP status
        status_code = 200
        if result.status == HealthStatus.DEGRADED:
            status_code = 200  # Still operational
        elif result.status == HealthStatus.UNHEALTHY:
            status_code = 503  # Service unavailable
        
        return {
            "status": result.status.value,
            "module": result.module_name,
            "timestamp": result.timestamp,
            "uptime_seconds": result.uptime_seconds,
            "checks": result.checks,
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    return health

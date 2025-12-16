"""
Health Check Implementation for Risk-Safety Service
SPRINT 3 - Part 2: Infrastructure Hardening

Add this to microservices/risk_safety/service.py
"""

# Add to imports:
from backend.core.health_contract import (
    ServiceHealth, DependencyHealth, DependencyStatus,
    check_redis_health
)
from datetime import datetime, timezone

# Add to __init__:
self._start_time = datetime.now(timezone.utc)

# Replace get_health() method with:
async def get_health(self) -> Dict[str, Any]:
    """Get service health status (standardized format)."""
    dependencies = {}
    
    # Check Redis (PolicyStore backend)
    if hasattr(self, 'policy_store') and hasattr(self.policy_store, 'redis_client'):
        dependencies["redis"] = await check_redis_health(self.policy_store.redis_client)
    else:
        dependencies["redis"] = DependencyHealth(
            status=DependencyStatus.DOWN,
            error="Redis/PolicyStore not initialized"
        )
    
    # Check EventBus
    if hasattr(self.event_bus, 'redis_client') and self.event_bus.redis_client:
        dependencies["eventbus"] = DependencyHealth(
            status=DependencyStatus.OK if self._running else DependencyStatus.DOWN
        )
    else:
        dependencies["eventbus"] = DependencyHealth(
            status=DependencyStatus.DOWN,
            error="EventBus not initialized"
        )
    
    # Check PostgreSQL (if used for ESS state)
    dependencies["postgres"] = DependencyHealth(
        status=DependencyStatus.NOT_APPLICABLE,
        details={"note": "ESS state stored in Redis"}
    )
    
    # Service-specific metrics
    metrics = {
        "ess_tripped": getattr(self.ess, 'is_tripped', False) if hasattr(self, 'ess') else False,
        "policies_loaded": len(getattr(self.policy_store, '_policies', {})) if hasattr(self, 'policy_store') else 0,
        "risk_checks_total": getattr(self, '_risk_checks_total', 0),
        "running": self._running
    }
    
    # Build standardized health response
    health = ServiceHealth.create(
        service_name="risk-safety-service",
        version=settings.VERSION,
        start_time=self._start_time,
        dependencies=dependencies,
        metrics=metrics
    )
    
    return health.to_dict()

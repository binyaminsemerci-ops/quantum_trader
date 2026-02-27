"""
Health Check Implementation for Execution Service
SPRINT 3 - Part 2: Infrastructure Hardening

Add this to microservices/execution/service.py
"""

# Add to imports:
from backend.core.health_contract import (
    ServiceHealth, DependencyHealth, DependencyStatus,
    check_redis_health, check_http_endpoint_health
)
from datetime import datetime, timezone

# Add to __init__:
self._start_time = datetime.now(timezone.utc)

# Replace get_health() method with:
async def get_health(self) -> Dict[str, Any]:
    """Get service health status (standardized format)."""
    dependencies = {}
    
    # Check Redis (via EventBus)
    if hasattr(self.event_bus, 'redis_client') and self.event_bus.redis_client:
        dependencies["redis"] = await check_redis_health(self.event_bus.redis_client)
    else:
        dependencies["redis"] = DependencyHealth(
            status=DependencyStatus.DOWN,
            error="Redis client not initialized"
        )
    
    # Check EventBus
    dependencies["eventbus"] = DependencyHealth(
        status=DependencyStatus.OK if self.event_bus and self._running else DependencyStatus.DOWN
    )
    
    # Check Binance API
    if self.binance_client:
        try:
            import time
            start = time.time()
            await self.binance_client.get_server_time()
            latency_ms = (time.time() - start) * 1000
            dependencies["binance_api"] = DependencyHealth(
                status=DependencyStatus.OK,
                latency_ms=latency_ms
            )
        except Exception as e:
            dependencies["binance_api"] = DependencyHealth(
                status=DependencyStatus.DOWN,
                error=str(e)[:100]
            )
    else:
        dependencies["binance_api"] = DependencyHealth(
            status=DependencyStatus.NOT_APPLICABLE
        )
    
    # Check TradeStore (SQLite)
    dependencies["tradestore"] = DependencyHealth(
        status=DependencyStatus.OK if hasattr(self, 'trade_store') else DependencyStatus.DOWN
    )
    
    # Service-specific metrics
    metrics = {
        "active_positions": len(getattr(self, '_active_positions', {})),
        "orders_executed_total": getattr(self, '_orders_executed', 0),
        "running": self._running
    }
    
    # Build standardized health response
    health = ServiceHealth.create(
        service_name="execution-service",
        version=settings.VERSION,
        start_time=self._start_time,
        dependencies=dependencies,
        metrics=metrics
    )
    
    return health.to_dict()

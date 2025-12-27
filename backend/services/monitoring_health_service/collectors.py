"""
Health Collectors for Monitoring Health Service

Collects health snapshots from:
- Other microservices via HTTP /health endpoints
- Infrastructure components (Redis, Postgres, Binance API)
- Internal metrics and EventBus status

Author: Quantum Trader AI Team
Date: December 4, 2025
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ServiceTarget:
    """Configuration for a service to monitor."""
    
    def __init__(self, name: str, url: str, critical: bool = True):
        self.name = name
        self.url = url
        self.critical = critical
    
    def __repr__(self) -> str:
        return f"ServiceTarget(name={self.name}, url={self.url}, critical={self.critical})"


class InfraTarget:
    """Configuration for infrastructure component to monitor."""
    
    def __init__(self, name: str, type: str, config: Dict[str, Any]):
        self.name = name
        self.type = type  # "redis", "postgres", "binance_api"
        self.config = config
    
    def __repr__(self) -> str:
        return f"InfraTarget(name={self.name}, type={self.type})"


class HealthCollector:
    """
    Collects health status from all services and infrastructure.
    
    Responsibilities:
    - Call /health endpoints on registered services
    - Ping Redis/Postgres/Binance
    - Measure latency for each check
    - Return structured snapshot
    """
    
    def __init__(
        self,
        service_targets: List[ServiceTarget],
        infra_targets: List[InfraTarget],
        http_client: httpx.AsyncClient,
        redis_client: Optional[Redis] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.service_targets = service_targets
        self.infra_targets = infra_targets
        self.http_client = http_client
        self.redis_client = redis_client
        self.logger = logger or logging.getLogger(__name__)
    
    async def collect_snapshot(self) -> Dict[str, Any]:
        """
        Collect health status from all targets.
        
        Returns:
            {
                "timestamp": "2025-12-04T10:30:00Z",
                "services": {
                    "service_name": {
                        "status": "OK" | "DEGRADED" | "DOWN",
                        "latency_ms": 45.3,
                        "details": {...},
                        "error": "..." (if failed)
                    }
                },
                "infra": {
                    "redis": {"status": "OK", "latency_ms": 2.1},
                    "postgres": {"status": "OK", "latency_ms": 5.4},
                    "binance_api": {"status": "OK", "latency_ms": 120.5}
                }
            }
        """
        self.logger.info("[HealthCollector] Starting health snapshot collection")
        
        # Collect in parallel
        service_results = await self._collect_services()
        infra_results = await self._collect_infrastructure()
        
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": service_results,
            "infra": infra_results,
        }
        
        # Count statuses
        total_services = len(service_results)
        down_services = sum(1 for s in service_results.values() if s["status"] == "DOWN")
        degraded_services = sum(1 for s in service_results.values() if s["status"] == "DEGRADED")
        
        self.logger.info(
            f"[HealthCollector] Snapshot complete: {total_services} services "
            f"({down_services} DOWN, {degraded_services} DEGRADED)"
        )
        
        return snapshot
    
    async def _collect_services(self) -> Dict[str, Dict[str, Any]]:
        """Collect health from all registered services."""
        tasks = [
            self._check_service(target)
            for target in self.service_targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        service_health = {}
        for target, result in zip(self.service_targets, results):
            if isinstance(result, Exception):
                service_health[target.name] = {
                    "status": "DOWN",
                    "latency_ms": None,
                    "error": str(result),
                    "critical": target.critical,
                }
            else:
                service_health[target.name] = result
        
        return service_health
    
    async def _check_service(self, target: ServiceTarget) -> Dict[str, Any]:
        """Check health of a single service via HTTP."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await self.http_client.get(target.url)
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if response.status_code == 200:
                status = "OK"
                try:
                    details = response.json()
                except Exception:
                    details = {"raw": response.text[:200]}
            elif 500 <= response.status_code < 600:
                status = "DOWN"
                details = {"status_code": response.status_code}
            else:
                status = "DEGRADED"
                details = {"status_code": response.status_code}
            
            return {
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "details": details,
                "critical": target.critical,
            }
        
        except httpx.TimeoutException:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return {
                "status": "DOWN",
                "latency_ms": round(latency_ms, 2),
                "error": "Request timeout",
                "critical": target.critical,
            }
        
        except httpx.ConnectError as e:
            return {
                "status": "DOWN",
                "latency_ms": None,
                "error": f"Connection failed: {str(e)}",
                "critical": target.critical,
            }
        
        except Exception as e:
            return {
                "status": "DOWN",
                "latency_ms": None,
                "error": f"Unexpected error: {str(e)}",
                "critical": target.critical,
            }
    
    async def _collect_infrastructure(self) -> Dict[str, Dict[str, Any]]:
        """Collect health from infrastructure components."""
        tasks = [
            self._check_infrastructure(target)
            for target in self.infra_targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        infra_health = {}
        for target, result in zip(self.infra_targets, results):
            if isinstance(result, Exception):
                infra_health[target.name] = {
                    "status": "DOWN",
                    "latency_ms": None,
                    "error": str(result),
                }
            else:
                infra_health[target.name] = result
        
        return infra_health
    
    async def _check_infrastructure(self, target: InfraTarget) -> Dict[str, Any]:
        """Check health of a single infrastructure component."""
        if target.type == "redis":
            return await self._check_redis(target)
        elif target.type == "postgres":
            return await self._check_postgres(target)
        elif target.type == "binance_api":
            return await self._check_binance_api(target)
        else:
            return {
                "status": "UNKNOWN",
                "error": f"Unknown infra type: {target.type}",
            }
    
    async def _check_redis(self, target: InfraTarget) -> Dict[str, Any]:
        """Ping Redis to check connectivity."""
        if not self.redis_client:
            return {
                "status": "DOWN",
                "error": "Redis client not initialized",
            }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            await self.redis_client.ping()
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "OK",
                "latency_ms": round(latency_ms, 2),
            }
        
        except Exception as e:
            return {
                "status": "DOWN",
                "latency_ms": None,
                "error": f"Redis ping failed: {str(e)}",
            }
    
    async def _check_postgres(self, target: InfraTarget) -> Dict[str, Any]:
        """Check Postgres connectivity."""
        # For now, we'll check via HTTP endpoint that queries DB
        # In production, you'd use asyncpg or psycopg3 here
        db_health_url = target.config.get("health_url")
        
        if not db_health_url:
            return {
                "status": "UNKNOWN",
                "error": "No health_url configured for Postgres",
            }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await self.http_client.get(db_health_url)
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if response.status_code == 200:
                return {
                    "status": "OK",
                    "latency_ms": round(latency_ms, 2),
                }
            else:
                return {
                    "status": "DEGRADED",
                    "latency_ms": round(latency_ms, 2),
                    "error": f"Status code: {response.status_code}",
                }
        
        except Exception as e:
            return {
                "status": "DOWN",
                "latency_ms": None,
                "error": f"Postgres check failed: {str(e)}",
            }
    
    async def _check_binance_api(self, target: InfraTarget) -> Dict[str, Any]:
        """Check Binance API reachability."""
        api_url = target.config.get("url", "https://api.binance.com/api/v3/ping")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await self.http_client.get(api_url)
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if response.status_code == 200:
                return {
                    "status": "OK",
                    "latency_ms": round(latency_ms, 2),
                }
            else:
                return {
                    "status": "DEGRADED",
                    "latency_ms": round(latency_ms, 2),
                    "error": f"Status code: {response.status_code}",
                }
        
        except Exception as e:
            return {
                "status": "DOWN",
                "latency_ms": None,
                "error": f"Binance API check failed: {str(e)}",
            }

"""
Main Entry Point for Monitoring Health Service

Initializes and runs the monitoring health service with periodic health checks
and event-driven alerting.

Author: Quantum Trader AI Team
Date: December 4, 2025
Sprint: 2 - Service #6

Usage:
    python -m backend.services.monitoring_health_service.main
"""

import asyncio
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Optional

from backend.core.event_bus import EventBus
from backend.core import configure_logging

from .dependencies import ServiceDependencies
from .collectors import HealthCollector, ServiceTarget, InfraTarget
from .aggregators import HealthAggregator
from .alerting import AlertManager
from .app import create_health_app

logger = logging.getLogger(__name__)


class MonitoringHealthService:
    """
    Main monitoring health service orchestrator.
    
    Responsibilities:
    - Initialize all components (EventBus, HTTP client, Redis)
    - Run periodic health checks (every 30-60 seconds)
    - Listen for critical events (ess.tripped)
    - Publish health.snapshot_updated events
    - Serve HTTP API for health status queries
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        check_interval_seconds: int = 60,
        service_name: str = "monitoring_health_service",
    ):
        self.redis_url = redis_url
        self.check_interval_seconds = check_interval_seconds
        self.service_name = service_name
        
        # Components
        self.dependencies: Optional[ServiceDependencies] = None
        self.event_bus: Optional[EventBus] = None
        self.collector: Optional[HealthCollector] = None
        self.aggregator: Optional[HealthAggregator] = None
        self.alert_manager: Optional[AlertManager] = None
        self.app = None
        
        # Control
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("[Monitoring Health] Initializing service...")
        
        # Initialize dependencies
        self.dependencies = ServiceDependencies(redis_url=self.redis_url)
        await self.dependencies.initialize()
        
        # Initialize EventBus
        if self.dependencies.is_redis_available():
            self.event_bus = EventBus(
                redis_client=self.dependencies.redis_client,
                service_name=self.service_name,
            )
            await self.event_bus.initialize()
            logger.info("[Monitoring Health] EventBus initialized")
        else:
            logger.warning("[Monitoring Health] Redis unavailable, EventBus disabled")
        
        # Define service targets (other microservices to monitor)
        service_targets = self._get_service_targets()
        
        # Define infrastructure targets
        infra_targets = self._get_infra_targets()
        
        # Initialize collector
        self.collector = HealthCollector(
            service_targets=service_targets,
            infra_targets=infra_targets,
            http_client=self.dependencies.http_client,
            redis_client=self.dependencies.redis_client,
        )
        logger.info(
            f"[Monitoring Health] Collector initialized: {len(service_targets)} services, "
            f"{len(infra_targets)} infra components"
        )
        
        # Initialize aggregator
        self.aggregator = HealthAggregator()
        logger.info("[Monitoring Health] Aggregator initialized")
        
        # Initialize alert manager
        self.alert_manager = AlertManager(event_bus=self.event_bus)
        logger.info("[Monitoring Health] Alert manager initialized")
        
        # Register event handlers
        if self.event_bus:
            self.event_bus.subscribe("ess.tripped", self._handle_ess_tripped)
            logger.info("[Monitoring Health] Subscribed to ess.tripped events")
        
        # Create FastAPI app
        self.app = create_health_app(
            collector=self.collector,
            aggregator=self.aggregator,
            alert_manager=self.alert_manager,
        )
        logger.info("[Monitoring Health] API app created")
        
        logger.info("[Monitoring Health] ✅ Service initialized successfully")
    
    def _get_service_targets(self) -> list[ServiceTarget]:
        """
        Get list of services to monitor.
        
        NOTE: Adjust these URLs based on your actual service deployment.
        """
        base_url = os.getenv("BASE_SERVICE_URL", "http://localhost:8000")
        
        return [
            # Main backend
            ServiceTarget(
                name="main_backend",
                url=f"{base_url}/health",
                critical=True,
            ),
            # Scheduler
            ServiceTarget(
                name="scheduler",
                url=f"{base_url}/health/scheduler",
                critical=True,
            ),
            # Risk Guard
            ServiceTarget(
                name="risk_guard",
                url=f"{base_url}/health/risk",
                critical=True,
            ),
            # AI System
            ServiceTarget(
                name="ai_system",
                url=f"{base_url}/health/ai",
                critical=True,
            ),
            # System Health Monitor
            ServiceTarget(
                name="system_health_monitor",
                url=f"{base_url}/health/system",
                critical=False,
            ),
            # Add more services as needed
        ]
    
    def _get_infra_targets(self) -> list[InfraTarget]:
        """Get list of infrastructure components to monitor."""
        return [
            # Redis
            InfraTarget(
                name="redis",
                type="redis",
                config={"url": self.redis_url},
            ),
            # Postgres (via health endpoint)
            InfraTarget(
                name="postgres",
                type="postgres",
                config={
                    "health_url": os.getenv(
                        "POSTGRES_HEALTH_URL",
                        "http://localhost:8000/health",  # Assumes main backend checks DB
                    ),
                },
            ),
            # Binance API
            InfraTarget(
                name="binance_api",
                type="binance_api",
                config={"url": "https://api.binance.com/api/v3/ping"},
            ),
        ]
    
    async def start(self) -> None:
        """Start the monitoring service."""
        if self._running:
            logger.warning("[Monitoring Health] Service already running")
            return
        
        self._running = True
        logger.info("[Monitoring Health] Starting service...")
        
        # Start EventBus consumer
        if self.event_bus:
            await self.event_bus.start()
            logger.info("[Monitoring Health] EventBus consumer started")
        
        # Start periodic health checks
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        logger.info(
            f"[Monitoring Health] Periodic health checks started "
            f"(interval: {self.check_interval_seconds}s)"
        )
        
        logger.info("[Monitoring Health] ✅ Service running")
    
    async def stop(self) -> None:
        """Stop the monitoring service."""
        if not self._running:
            return
        
        self._running = False
        logger.info("[Monitoring Health] Stopping service...")
        
        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop EventBus
        if self.event_bus:
            await self.event_bus.stop()
            logger.info("[Monitoring Health] EventBus stopped")
        
        # Cleanup dependencies
        if self.dependencies:
            await self.dependencies.cleanup()
        
        logger.info("[Monitoring Health] ✅ Service stopped")
    
    async def _periodic_health_check(self) -> None:
        """Run health checks periodically."""
        logger.info("[Monitoring Health] Periodic health check loop started")
        
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                
                if not self._running:
                    break
                
                logger.info("[Monitoring Health] Running periodic health check...")
                
                # Collect snapshot
                snapshot = await self.collector.collect_snapshot()
                
                # Aggregate
                aggregated = self.aggregator.aggregate(snapshot)
                
                # Process alerts
                if self.alert_manager:
                    await self.alert_manager.process_health(aggregated)
                
                # Publish snapshot event
                if self.event_bus:
                    await self.event_bus.publish(
                        "health.snapshot_updated",
                        aggregated.to_dict(),
                    )
                
                logger.info(
                    f"[Monitoring Health] Health check complete: status={aggregated.status.value}"
                )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Monitoring Health] Health check failed: {e}", exc_info=True)
    
    async def _handle_ess_tripped(self, event_data: dict) -> None:
        """Handle ess.tripped event."""
        logger.critical(f"[Monitoring Health] ESS TRIPPED event received: {event_data}")
        
        if self.alert_manager:
            await self.alert_manager.process_ess_tripped(event_data)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point."""
    # Configure logging
    configure_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format="json" if os.getenv("LOG_FORMAT") == "json" else "text",
    )
    
    # Create service
    service = MonitoringHealthService(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        check_interval_seconds=int(os.getenv("HEALTH_CHECK_INTERVAL", "60")),
    )
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("[Monitoring Health] Received shutdown signal")
        asyncio.create_task(service.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Initialize and start
        await service.initialize()
        await service.start()
        
        # Keep running
        logger.info("[Monitoring Health] Service ready, press Ctrl+C to stop")
        while service._running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("[Monitoring Health] Keyboard interrupt received")
    except Exception as e:
        logger.error(f"[Monitoring Health] Fatal error: {e}", exc_info=True)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())

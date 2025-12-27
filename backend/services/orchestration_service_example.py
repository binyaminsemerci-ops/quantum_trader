"""Example orchestration service implementation.

This module demonstrates how to integrate and run the AI Orchestration Layer
in a production environment.
"""

import asyncio
import logging
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.ai_orchestrator import AI_CEO
from backend.ai_risk import AI_RiskOfficer
from backend.ai_strategy import AI_StrategyOfficer
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.federation import FederatedEngine

logger = logging.getLogger(__name__)


class OrchestrationService:
    """
    Orchestration Service - Manages all AI agents and federation layer.
    
    This service can be run standalone or integrated into analytics-os-service.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        service_name: str = "orchestration_service",
    ):
        """Initialize Orchestration Service."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.service_name = service_name
        
        # Components
        self.redis_client: Optional[Redis] = None
        self.event_bus: Optional[EventBus] = None
        self.policy_store: Optional[PolicyStore] = None
        
        # AI Agents
        self.ai_ceo: Optional[AI_CEO] = None
        self.ai_risk_officer: Optional[AI_RiskOfficer] = None
        self.ai_strategy_officer: Optional[AI_StrategyOfficer] = None
        self.federated_engine: Optional[FederatedEngine] = None
        
        # State
        self._initialized = False
        self._running = False
        
        logger.info(f"OrchestrationService created: service_name={service_name}")
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            logger.warning("OrchestrationService already initialized")
            return
        
        logger.info("Initializing OrchestrationService...")
        
        # Initialize Redis
        self.redis_client = Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=False,
        )
        
        await self.redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # Initialize EventBus
        self.event_bus = EventBus(
            redis_client=self.redis_client,
            service_name=self.service_name,
        )
        await self.event_bus.initialize()
        logger.info("✅ EventBus initialized")
        
        # Initialize PolicyStore
        self.policy_store = PolicyStore(
            redis_client=self.redis_client,
            event_bus=self.event_bus,
        )
        await self.policy_store.initialize()
        logger.info("✅ PolicyStore initialized")
        
        # Initialize AI CEO
        self.ai_ceo = AI_CEO(
            redis_client=self.redis_client,
            event_bus=self.event_bus,
            policy_store=self.policy_store,
            decision_interval=30.0,
        )
        await self.ai_ceo.initialize()
        logger.info("✅ AI CEO initialized")
        
        # Initialize AI Risk Officer
        self.ai_risk_officer = AI_RiskOfficer(
            redis_client=self.redis_client,
            event_bus=self.event_bus,
            policy_store=self.policy_store,
            assessment_interval=30.0,
        )
        await self.ai_risk_officer.initialize()
        logger.info("✅ AI Risk Officer initialized")
        
        # Initialize AI Strategy Officer
        self.ai_strategy_officer = AI_StrategyOfficer(
            redis_client=self.redis_client,
            event_bus=self.event_bus,
            policy_store=self.policy_store,
            analysis_interval=60.0,
        )
        await self.ai_strategy_officer.initialize()
        logger.info("✅ AI Strategy Officer initialized")
        
        # Initialize Federated Engine
        self.federated_engine = FederatedEngine(
            redis_client=self.redis_client,
            event_bus=self.event_bus,
            update_interval=15.0,
        )
        await self.federated_engine.initialize()
        logger.info("✅ Federated Engine initialized")
        
        self._initialized = True
        logger.info("🎉 OrchestrationService initialization complete")
    
    async def start(self) -> None:
        """Start all AI agents and federation layer."""
        if not self._initialized:
            raise RuntimeError("OrchestrationService not initialized")
        
        if self._running:
            logger.warning("OrchestrationService already running")
            return
        
        logger.info("Starting OrchestrationService...")
        
        # Start AI CEO
        await self.ai_ceo.start()
        logger.info("✅ AI CEO started")
        
        # Start AI Risk Officer
        await self.ai_risk_officer.start()
        logger.info("✅ AI Risk Officer started")
        
        # Start AI Strategy Officer
        await self.ai_strategy_officer.start()
        logger.info("✅ AI Strategy Officer started")
        
        # Start Federated Engine
        await self.federated_engine.start()
        logger.info("✅ Federated Engine started")
        
        self._running = True
        logger.info("🚀 OrchestrationService started successfully")
    
    async def stop(self) -> None:
        """Stop all components gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping OrchestrationService...")
        
        # Stop agents
        if self.ai_ceo:
            await self.ai_ceo.stop()
        if self.ai_risk_officer:
            await self.ai_risk_officer.stop()
        if self.ai_strategy_officer:
            await self.ai_strategy_officer.stop()
        if self.federated_engine:
            await self.federated_engine.stop()
        
        # Close Redis
        if self.redis_client:
            await self.redis_client.close()
        
        self._running = False
        logger.info("✅ OrchestrationService stopped")
    
    async def get_status(self) -> dict:
        """Get comprehensive status of all components."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if self._running else "initialized",
            "service_name": self.service_name,
            "components": {
                "ai_ceo": await self.ai_ceo.get_status() if self.ai_ceo else None,
                "ai_risk_officer": await self.ai_risk_officer.get_status() if self.ai_risk_officer else None,
                "ai_strategy_officer": await self.ai_strategy_officer.get_status() if self.ai_strategy_officer else None,
                "federated_engine": await self.federated_engine.get_status() if self.federated_engine else None,
            },
        }
    
    def get_global_state(self):
        """Get current global state from federation engine."""
        if not self.federated_engine:
            return None
        return self.federated_engine.get_current_global_state()


async def main():
    """Main entry point for standalone orchestration service."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create and start service
    service = OrchestrationService(
        redis_host="localhost",
        redis_port=6379,
        service_name="orchestration_service",
    )
    
    try:
        # Initialize
        await service.initialize()
        
        # Start
        await service.start()
        
        # Print status
        status = await service.get_status()
        logger.info(f"Service status: {status}")
        
        # Keep running
        logger.info("Orchestration service running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
            
            # Periodic status log
            global_state = service.get_global_state()
            if global_state:
                logger.info(
                    f"Global State: mode={global_state.global_mode}, "
                    f"risk={global_state.risk_level}, "
                    f"completeness={global_state.data_completeness:.2f}"
                )
    
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        await service.stop()
        logger.info("Orchestration service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

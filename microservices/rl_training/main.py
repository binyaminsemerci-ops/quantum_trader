"""
Main entry point for RL Training / CLM / Shadow Models Service

Service: rl-training-service (port 8005)

Responsibilities:
- Train RL (PPO) and supervised ML models (XGB, LGBM, etc.)
- Continuous learning management
- Shadow model testing and promotion
- Drift detection and monitoring
"""
import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
import uvicorn

from microservices.rl_training.config import settings
from microservices.rl_training.api import router
from microservices.rl_training.training_daemon import RLTrainingDaemon
from microservices.rl_training.clm import ContinuousLearningManager
from microservices.rl_training.shadow_models import ShadowModelManager
from microservices.rl_training.drift_detection import DriftDetector
from microservices.rl_training.handlers import EventHandlers, setup_event_subscriptions
from microservices.rl_training.scheduler import TrainingScheduler
from microservices.rl_training.dependencies import create_fake_dependencies


# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instance
service_instance: Optional["RLTrainingService"] = None
service_start_time: datetime = datetime.now(timezone.utc)


class RLTrainingService:
    """
    RL Training Service
    
    Main service orchestrator for:
    - RL training (PPO)
    - Continuous learning (XGB, LGBM, etc.)
    - Shadow model testing
    - Drift detection
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        
        # Dependencies (using fakes for now)
        (
            self.policy_store,
            self.data_source,
            self.model_registry,
            self.event_bus
        ) = create_fake_dependencies(config)
        
        # Core components
        self.training_daemon = RLTrainingDaemon(
            policy_store=self.policy_store,
            data_source=self.data_source,
            model_registry=self.model_registry,
            event_bus=self.event_bus,
            config=self.config,
            logger_instance=self.logger
        )
        
        self.clm = ContinuousLearningManager(
            training_daemon=self.training_daemon,
            config=self.config,
            logger_instance=self.logger
        )
        
        self.shadow_manager = ShadowModelManager(
            model_registry=self.model_registry,
            event_bus=self.event_bus,
            config=self.config,
            logger_instance=self.logger
        )
        
        self.drift_detector = DriftDetector(
            event_bus=self.event_bus,
            config=self.config,
            logger_instance=self.logger
        )
        
        # Event handlers
        self.event_handlers = EventHandlers(
            training_daemon=self.training_daemon,
            clm=self.clm,
            drift_detector=self.drift_detector,
            config=self.config,
            logger_instance=self.logger
        )
        
        # Scheduler
        self.scheduler = TrainingScheduler(
            training_daemon=self.training_daemon,
            clm=self.clm,
            drift_detector=self.drift_detector,
            config=self.config,
            logger_instance=self.logger
        )
        
        self._running = False
    
    async def start(self):
        """Start service"""
        if self._running:
            self.logger.debug("[RLTrainingService] Already running")
            return
        
        self._running = True
        
        self.logger.info(
            "[RLTrainingService] Starting service",
            extra={
                "service": settings.SERVICE_NAME,
                "version": settings.SERVICE_VERSION,
                "port": settings.PORT
            }
        )
        
        # Setup event subscriptions
        setup_event_subscriptions(self.event_bus, self.event_handlers)
        
        # Start scheduler
        await self.scheduler.start()
        
        self.logger.info("[RLTrainingService] Service started successfully")
    
    async def stop(self):
        """Stop service"""
        if not self._running:
            return
        
        self._running = False
        
        self.logger.info("[RLTrainingService] Stopping service")
        
        # Stop scheduler
        await self.scheduler.stop()
        
        self.logger.info("[RLTrainingService] Service stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global service_instance
    
    # Startup
    logger.info(
        f"[Lifespan] Starting {settings.SERVICE_NAME} service v{settings.SERVICE_VERSION}"
    )
    
    service_instance = RLTrainingService(settings)
    await service_instance.start()
    
    # Store in app state for dependency injection
    app.state.training_daemon = service_instance.training_daemon
    app.state.clm = service_instance.clm
    app.state.shadow_manager = service_instance.shadow_manager
    app.state.drift_detector = service_instance.drift_detector
    app.state.event_bus = service_instance.event_bus
    
    logger.info("[Lifespan] Service startup complete")
    
    yield
    
    # Shutdown
    logger.info("[Lifespan] Shutting down service")
    await service_instance.stop()
    logger.info("[Lifespan] Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RL Training / CLM / Shadow Models Service",
    description="Training orchestration, continuous learning, shadow testing, drift detection",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

# Include API router
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/api/training/health",
            "training_history": "/api/training/jobs/history",
            "clm_status": "/api/training/clm/status",
            "shadow_models": "/api/training/shadow/models",
            "drift_history": "/api/training/drift/history"
        }
    }


def handle_shutdown_signal(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"[Signal] Received signal {signum}, initiating graceful shutdown")
    # FastAPI lifespan will handle cleanup
    raise KeyboardInterrupt


# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown_signal)
signal.signal(signal.SIGINT, handle_shutdown_signal)


if __name__ == "__main__":
    uvicorn.run(
        "microservices.rl_training.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

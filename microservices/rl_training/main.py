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
import json
import logging
import os
import signal
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Response
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST
import uvicorn
import redis

from microservices.rl_training.config import settings
from microservices.rl_training.api import router
from microservices.rl_training.training_daemon import RLTrainingDaemon
from microservices.rl_training.clm import ContinuousLearningManager
from microservices.rl_training.shadow_models import ShadowModelManager
from microservices.rl_training.drift_detection import DriftDetector
from microservices.rl_training.handlers import EventHandlers, setup_event_subscriptions
from microservices.rl_training.scheduler import TrainingScheduler
from microservices.rl_training.dependencies import create_fake_dependencies


# [EPIC-OBS-001] Initialize observability (tracing, metrics, structured logging)
try:
    from backend.infra.observability import (
        init_observability,
        get_logger,
        instrument_fastapi,
        add_metrics_middleware,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    # Fallback to basic logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Initialize observability at module level (before service starts)
if OBSERVABILITY_AVAILABLE:
    init_observability(
        service_name="rl-training",
        log_level=settings.LOG_LEVEL,
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
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

        # Redis (reward stream + heartbeat)
        self.redis = redis.Redis(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            db=self.config.REDIS_DB,
            decode_responses=True
        )
        self.reward_stream_key = "quantum:stream:rl_rewards"
        self.heartbeat_key = "quantum:svc:rl_trainer:heartbeat"
        self.heartbeat_ttl = int(os.getenv("RL_TRAINER_HEARTBEAT_TTL", "30"))
        self._consumer_thread: Optional[threading.Thread] = None
        self._consumer_running = False
        self._last_reward_id = "0-0"
        self._model_update_dir = Path(
            os.getenv("RL_TRAINER_MODEL_UPDATE_DIR", self.config.MODEL_SAVE_DIR)
        )
        self._model_update_dir.mkdir(parents=True, exist_ok=True)
        
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

        # Start reward consumer
        self._start_reward_consumer()
        
        self.logger.info("[RLTrainingService] Service started successfully")
    
    async def stop(self):
        """Stop service"""
        if not self._running:
            return
        
        self._running = False

        # Stop reward consumer
        self._stop_reward_consumer()
        
        self.logger.info("[RLTrainingService] Stopping service")
        
        # Stop scheduler
        await self.scheduler.stop()
        
        self.logger.info("[RLTrainingService] Service stopped")

    def _start_reward_consumer(self):
        if self._consumer_running:
            return

        self._consumer_running = True
        self._consumer_thread = threading.Thread(
            target=self._reward_consumer_loop,
            name="rl_reward_consumer",
            daemon=True
        )
        self._consumer_thread.start()
        self.logger.info("[RewardConsumer] Started")

    def _stop_reward_consumer(self):
        self._consumer_running = False
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5)
        self.logger.info("[RewardConsumer] Stopped")

    def _reward_consumer_loop(self):
        """Consume rewards from Redis stream and write policy update artifacts."""
        while self._consumer_running:
            try:
                # Heartbeat
                try:
                    self.redis.set(self.heartbeat_key, str(int(time.time())), ex=self.heartbeat_ttl)
                except Exception as e:
                    self.logger.warning(f"[RewardConsumer] Failed to set heartbeat: {e}")

                messages = self.redis.xread(
                    {self.reward_stream_key: self._last_reward_id},
                    count=10,
                    block=5000
                )

                if messages:
                    for stream_key, message_list in messages:
                        for message_id, data in message_list:
                            self._last_reward_id = message_id

                            update_payload = {
                                "reward_id": message_id,
                                "reward": data,
                                "updated_at": datetime.now(timezone.utc).isoformat()
                            }
                            update_path = self._model_update_dir / f"rl_policy_update_{message_id.replace(':', '_')}.json"
                            try:
                                update_path.write_text(json.dumps(update_payload, indent=2))
                                self.redis.set(
                                    "quantum:rl:policy:last_update",
                                    update_payload["updated_at"],
                                    ex=86400
                                )
                                self.logger.info(
                                    f"[RewardConsumer] Consumed reward {message_id} -> {update_path}"
                                )
                            except Exception as e:
                                self.logger.error(f"[RewardConsumer] Failed to write update: {e}")

            except Exception as e:
                self.logger.error(f"[RewardConsumer] Error: {e}")
                time.sleep(1)


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

# [EPIC-OBS-001] Instrument FastAPI with tracing & metrics
if OBSERVABILITY_AVAILABLE:
    instrument_fastapi(app)
    add_metrics_middleware(app)

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


@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe."""
    if service_instance is None:
        return {"status": "not_ready", "ready": False}
    # Check if service has a public running status
    is_running = getattr(service_instance, 'is_running', lambda: getattr(service_instance, '_running', False))
    if callable(is_running):
        ready = is_running()
    else:
        ready = is_running
    return {"status": "ready" if ready else "not_ready", "ready": ready}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


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

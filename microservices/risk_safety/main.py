"""
Risk & Safety Service - Main Entry Point

Responsibilities:
- Emergency Stop System (ESS)
- PolicyStore (Single Source of Truth)
- Risk limit enforcement
- Safety validation

Port: 8003
Events: ess.tripped, ess.state.changed, policy.updated, risk.limit.exceeded
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST

from microservices.risk_safety.service import RiskSafetyService
from microservices.risk_safety.api import router
from microservices.risk_safety.config import settings

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
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/risk_safety_service.log')
        ]
    )

# Initialize observability at module level (before service starts)
if OBSERVABILITY_AVAILABLE:
    # Use getattr to safely access LOG_LEVEL, default to INFO if not present
    log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
    init_observability(
        service_name="risk-safety",
        log_level=log_level,
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

# Global service instance
service: RiskSafetyService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global service
    
    # Startup
    logger.info("=" * 60)
    logger.info("RISK & SAFETY SERVICE STARTING")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Port: {settings.PORT}")
    logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info("=" * 60)
    
    try:
        # Initialize service
        service = RiskSafetyService()
        await service.start()
        
        # Register shutdown handler
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s))
            )
        
        logger.info("✅ Risk & Safety Service READY")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Risk & Safety Service SHUTTING DOWN...")
        if service:
            await service.stop()
        logger.info("✅ Risk & Safety Service STOPPED")


async def shutdown(sig):
    """Graceful shutdown on signal."""
    logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
    # FastAPI will call lifespan's finally block


# Create FastAPI app
app = FastAPI(
    title="Risk & Safety Service",
    version=settings.VERSION,
    lifespan=lifespan
)

# [EPIC-OBS-001] Instrument FastAPI with tracing & metrics
if OBSERVABILITY_AVAILABLE:
    instrument_fastapi(app)
    add_metrics_middleware(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if service is None:
        return {"status": "starting", "healthy": False}
    
    return await service.get_health()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "risk-safety",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe."""
    if service is None:
        return {"status": "starting", "ready": False}
    
    health = await service.get_health()
    return {"status": "ready" if health.get("healthy", False) else "not_ready", "ready": health.get("healthy", False)}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
        reload=False  # Disable in production
    )

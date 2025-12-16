"""
AI Engine Service - Main Entry Point

Responsibilities:
- AI model inference (XGBoost, LightGBM, N-HiTS, PatchTST)
- Ensemble voting and signal aggregation
- Meta-strategy selection (RL-based)
- RL position sizing
- Market regime detection
- Memory state management
- Trade intent generation

Port: 8001
Events IN: market.tick, market.klines, trade.closed, policy.updated
Events OUT: ai.decision.made, ai.signal_generated, strategy.selected, sizing.decided, trade.intent
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST

from .service import AIEngineService
from . import api
from .config import settings

# [EPIC-OBS-001] Initialize observability (tracing, metrics, structured logging)
try:
    from backend.infra.observability import (
        init_observability, 
        get_logger, 
        get_tracer, 
        instrument_fastapi,
        add_metrics_middleware,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    # Fallback to basic logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{settings.LOG_DIR}/ai_engine_service.log')
        ]
    )

# Initialize observability at module level (before service starts)
if OBSERVABILITY_AVAILABLE:
    init_observability(
        service_name="ai-engine",
        log_level=settings.LOG_LEVEL,
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

# Global service instance
service: AIEngineService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global service
    
    # Startup
    logger.info("=" * 60)
    logger.info("🤖 AI ENGINE SERVICE STARTING")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Port: {settings.PORT}")
    logger.info(f"Ensemble Models: {settings.ENSEMBLE_MODELS}")
    logger.info("=" * 60)
    
    try:
        service = AIEngineService()
        await service.start()
        logger.info("✅ AI Engine Service STARTED")
        
        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 AI ENGINE SERVICE SHUTTING DOWN...")
        if service:
            await service.stop()
        logger.info("✅ AI Engine Service STOPPED")


async def shutdown(sig):
    """Graceful shutdown on signal."""
    logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
    # FastAPI will call lifespan's finally block


# Create FastAPI app
app = FastAPI(
    title="AI Engine Service",
    version=settings.VERSION,
    description="AI model inference, ensemble voting, meta-strategy selection, and RL position sizing",
    lifespan=lifespan
)

# [EPIC-OBS-001] Instrument FastAPI with OpenTelemetry tracing
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

# Include API router
app.include_router(api.router)

# Inject service instance into API module
@app.on_event("startup")
async def inject_service():
    """Inject service instance into API module."""
    api.set_service(service)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ai-engine",
        "version": settings.VERSION,
        "status": "running" if service and service._running else "stopped",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Legacy health check endpoint (backward compatible).
    
    Returns full health status. Use /health/ready for K8s readiness probes.
    """
    if not service:
        return {"healthy": False, "error": "Service not initialized"}
    
    return await service.get_health()


@app.get("/health/live", tags=["health"])
async def health_live():
    """
    Liveness probe endpoint.
    
    Returns 200 if the process is alive and responsive.
    Used by Kubernetes/Docker to detect if container needs restart.
    """
    return {
        "status": "ok",
        "service": "ai-engine",
        "version": settings.VERSION,
    }


@app.get("/health/ready", tags=["health"])
async def health_ready():
    """
    Readiness probe endpoint.
    
    Returns 200 if service is ready to accept traffic (dependencies healthy).
    Returns 503 if service is not ready (still initializing or deps down).
    """
    if not service or not service._running:
        return Response(
            content='{"status": "not_ready", "reason": "Service not initialized"}',
            status_code=503,
            media_type="application/json"
        )
    
    # Get full health status from service
    health = await service.get_health()
    
    # If service reports unhealthy, return 503
    if not health.get("healthy", False):
        return Response(
            content=f'{{"status": "not_ready", "health": {health}}}',
            status_code=503,
            media_type="application/json"
        )
    
    return {
        "status": "ready",
        "service": "ai-engine",
        "healthy": True,
    }


@app.get("/metrics", tags=["observability"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all service metrics in Prometheus text format for scraping.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )

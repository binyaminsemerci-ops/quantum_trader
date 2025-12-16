"""
Execution Service - Main Entry Point

Responsibilities:
- Order execution (entry + exit)
- Position monitoring and TP/SL management
- Trade lifecycle tracking
- Binance API integration with rate limiting

Port: 8002
Events: order.placed, order.filled, order.failed, trade.opened, trade.closed, position.updated
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST

from .service import ExecutionService
from . import api
from .config import settings

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
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{settings.LOG_DIR}/execution_service.log')
        ]
    )

# Initialize observability at module level
if OBSERVABILITY_AVAILABLE:
    init_observability(
        service_name="execution",
        log_level=settings.LOG_LEVEL,
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

# Global service instance
service: ExecutionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global service
    
    # Startup
    logger.info("=" * 60)
    logger.info("EXECUTION SERVICE STARTING")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Port: {settings.PORT}")
    logger.info(f"Environment: {'TESTNET' if settings.USE_BINANCE_TESTNET else 'MAINNET'}")
    logger.info("=" * 60)
    
    try:
        # Initialize service
        service = ExecutionService()
        await service.start()
        
        # Register shutdown handler
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s))
            )
        
        logger.info("✅ Execution Service READY")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Execution Service SHUTTING DOWN...")
        if service:
            await service.stop()
        logger.info("✅ Execution Service STOPPED")


async def shutdown(sig):
    """Graceful shutdown on signal."""
    logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
    # FastAPI will call lifespan's finally block


# Create FastAPI app
app = FastAPI(
    title="Execution Service",
    version=settings.VERSION,
    description="Order execution, position monitoring, and trade lifecycle management",
    lifespan=lifespan
)

# [EPIC-OBS-001] Instrument FastAPI with observability
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
@app.get("/health")
async def health_check():
    """
    Legacy health check endpoint (backward compatible).
    
    Returns full health status. Use /health/ready for K8s readiness probes.
    """
    if service is None:
        return {"status": "starting", "healthy": False}
    
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
        "service": "execution",
        "version": settings.VERSION,
    }


@app.get("/health/ready", tags=["health"])
async def health_ready():
    """
    Readiness probe endpoint.
    
    Returns 200 if service is ready to accept traffic (dependencies healthy).
    Returns 503 if service is not ready (still initializing or deps down).
    """
    if service is None:
        return Response(
            content='{"status": "not_ready", "reason": "Service not initialized"}',
            status_code=503,
            media_type="application/json"
        )
    
    # Get full health status from service
    health = await service.get_health()
    
    # If service reports unhealthy, return 503
    if not health.get("healthy", True):  # Default to healthy if key missing
        return Response(
            content=f'{{"status": "not_ready", "health": {health}}}',
            status_code=503,
            media_type="application/json"
        )
    
    return {
        "status": "ready",
        "service": "execution",
        "healthy": True,
    }


@app.get("/metrics", tags=["observability"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all service metrics in Prometheus text format for scraping.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "execution",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }eturn {
        "service": "execution",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False  # Disable in production
    )

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

from .service_v2 import ExecutionService
from . import api
from .config import settings

# [EPIC-OBS-001] Initialize observability (tracing, metrics, structured logging)
# CRITICAL: Always ensure logging works, even if observability fails
import os

# Ensure log directory exists
try:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARNING] Could not create log directory {settings.LOG_DIR}: {e}")

# Try observability first
try:
    from backend.infra.observability import (
        init_observability,
        get_logger,
        instrument_fastapi,
        add_metrics_middleware,
    )
    OBSERVABILITY_AVAILABLE = True
    
    # Initialize observability
    init_observability(
        service_name="execution",
        log_level=settings.LOG_LEVEL,
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
    
    # IMPORTANT: Also add stdout handler to ensure we see logs in docker logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(console_handler)
    
except Exception as e:
    print(f"[WARNING] Observability init failed: {e}, using basic logging")
    OBSERVABILITY_AVAILABLE = False
    
    # Fallback to basic logging (ALWAYS works)
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Try to add file handler, but don't crash if it fails
    try:
        handlers.append(logging.FileHandler(f'{settings.LOG_DIR}/execution_service.log'))
    except Exception as file_err:
        print(f"[WARNING] Could not create file handler: {file_err}")
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        handlers=handlers,
        force=True  # Override any existing config
    )
    logger = logging.getLogger(__name__)

# Global service instance
service: ExecutionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global service
    
    # Startup
    print("[DEBUG] Lifespan startup CALLED")
    try:
        print(f"[DEBUG] Settings loaded: VERSION={settings.VERSION}")
        print("[DEBUG] About to call logger.info()")
        logger.info("=" * 60)
        logger.info("EXECUTION SERVICE STARTING")
        logger.info(f"Version: {settings.VERSION}")
        logger.info(f"Port: {settings.PORT}")
        testnet_mode = getattr(settings, 'USE_BINANCE_TESTNET', False)
        logger.info(f"Environment: {'TESTNET' if testnet_mode else 'MAINNET'}")
        logger.info("=" * 60)
        print("[DEBUG] Logger.info() calls completed")
        
        # Initialize service
        print("[DEBUG] Initializing ExecutionService...")
        logger.info("[EXECUTION] Initializing ExecutionService...")
        service = ExecutionService(settings)
        print("[DEBUG] ExecutionService created, calling start()...")
        logger.info("[EXECUTION] service.start() INVOKED")
        await service.start()
        print("[DEBUG] service.start() completed!")
        logger.info("[EXECUTION] ✅ service.start() COMPLETED - Consumer loop running")
        
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
        return {"status": "starting", "healthy": False, "consumer_started": False}
    
    health_data = await service.get_health()
    
    # Add consumer status (convert ServiceHealth to dict if needed)
    if hasattr(health_data, 'dict'):
        health_dict = health_data.dict()
    elif hasattr(health_data, '__dict__'):
        health_dict = health_data.__dict__
    else:
        health_dict = dict(health_data) if isinstance(health_data, dict) else {}
    
    consumer_started = service._running and service.event_bus is not None and service.event_bus._running
    health_dict["consumer_started"] = consumer_started
    health_dict["event_bus_running"] = service.event_bus._running if service.event_bus else False
    
    return health_dict


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

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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from microservices.risk_safety.service import RiskSafetyService
from microservices.risk_safety.api import router
from microservices.risk_safety.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/risk_safety_service.log')
    ]
)
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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
        reload=False  # Disable in production
    )

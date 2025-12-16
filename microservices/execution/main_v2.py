"""
Execution Service - Main Entry Point V2

Clean microservice with NO monolith dependencies.

Port: 8002
Events: 
  - Consumes: trade.intent (from AI Engine)
  - Publishes: execution.result (order results)
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .service_v2 import ExecutionService
from .config import settings
from .models import ServiceHealth

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Global service instance
service: Optional[ExecutionService] = None


async def shutdown(sig):
    """Graceful shutdown handler"""
    logger.info(f"Received signal {sig.name}, shutting down...")
    if service:
        await service.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global service
    
    # Startup
    logger.info("=" * 60)
    logger.info("EXECUTION SERVICE V2 STARTING")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Port: {settings.PORT}")
    logger.info(f"Mode: {settings.EXECUTION_MODE}")
    logger.info("=" * 60)
    
    try:
        # Initialize service
        service = ExecutionService(settings)
        await service.start()
        
        # Register shutdown handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s))
            )
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down...")
        if service:
            await service.stop()
        logger.info("✅ Stopped")


# Create FastAPI app
app = FastAPI(
    title="Execution Service",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=ServiceHealth)
async def health():
    """Health check endpoint"""
    if not service:
        return ServiceHealth(
            service="execution",
            status="STARTING",
            version=settings.VERSION,
            components=[],
            mode=settings.EXECUTION_MODE
        )
    
    return await service.get_health()


@app.get("/positions")
async def get_positions():
    """Get active positions"""
    if not service:
        return {"error": "Service not initialized"}
    
    positions = await service.get_positions()
    return {
        "count": len(positions),
        "positions": positions
    }


@app.get("/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    if not service:
        return {"error": "Service not initialized"}
    
    trades = await service.get_trades(limit)
    return {
        "count": len(trades),
        "trades": trades
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "execution",
        "version": settings.VERSION,
        "mode": settings.EXECUTION_MODE,
        "status": "running" if service and service._running else "stopped"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

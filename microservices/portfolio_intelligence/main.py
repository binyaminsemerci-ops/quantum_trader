"""
Portfolio Intelligence Service - Main Entry Point
"""
import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from microservices.portfolio_intelligence.config import settings
from microservices.portfolio_intelligence.service import PortfolioIntelligenceService
from microservices.portfolio_intelligence import api

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{settings.LOG_DIR}/portfolio_intelligence.log")
    ]
)

logger = logging.getLogger(__name__)

# Global service instance
service: PortfolioIntelligenceService = None


def get_service() -> PortfolioIntelligenceService:
    """Get global service instance."""
    if service is None:
        logger.warning("[PORTFOLIO-INTELLIGENCE] Service not yet initialized, returning None")
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global service
    
    # Startup
    logger.info("=" * 80)
    logger.info(f"PORTFOLIO INTELLIGENCE SERVICE v{settings.VERSION}")
    logger.info("=" * 80)
    
    service = PortfolioIntelligenceService()
    await service.start()
    
    # Register service instance with API module
    from microservices.portfolio_intelligence import api as api_module
    api_module.set_service_instance(service)
    
    logger.info("[PORTFOLIO-INTELLIGENCE] Service instance ready for API")
    
    # Setup signal handlers for graceful shutdown (skip on Windows)
    import sys
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    
    yield
    
    # Shutdown
    await service.stop()
    logger.info("[PORTFOLIO-INTELLIGENCE] Service shutdown complete")


async def shutdown():
    """Graceful shutdown handler."""
    logger.info("[PORTFOLIO-INTELLIGENCE] Shutdown signal received")
    if service:
        await service.stop()


# Create FastAPI app
app = FastAPI(
    title="Portfolio Intelligence Service",
    description="Manages portfolio state, PnL, exposure, and drawdown metrics",
    version=settings.VERSION,
    lifespan=lifespan
)

# Include API routes
app.include_router(api.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "portfolio-intelligence",
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring systems."""
    from datetime import datetime, timezone
    
    # Get service status
    service_status = "healthy" if service and service._running else "starting"
    
    health_data = {
        "service": "portfolio-intelligence",
        "status": service_status,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add operational metrics if service is running
    if service and service._running:
        if hasattr(service, '_current_snapshot') and service._current_snapshot:
            health_data["positions_count"] = len(service._current_snapshot.positions)
            # Check what attributes are actually available
            if hasattr(service._current_snapshot, 'total_realized_pnl'):
                health_data["total_realized_pnl"] = service._current_snapshot.total_realized_pnl
            if hasattr(service._current_snapshot, 'total_unrealized_pnl'):
                health_data["total_unrealized_pnl"] = service._current_snapshot.total_unrealized_pnl
    
    return health_data


def get_service() -> PortfolioIntelligenceService:
    """Get global service instance."""
    return service


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)

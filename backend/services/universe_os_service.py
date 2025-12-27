"""
Universe OS Service
Dynamic symbol selection and filtering
"""
import asyncio
import logging
from fastapi import FastAPI
from backend.services.universe_manager import UniverseManager
from backend.services.common.health_check import HealthChecker, create_health_endpoint
from backend.services.common.feature_flags import is_enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Universe OS", version="1.0.0")

# Initialize
universe_manager = None
health_checker = HealthChecker("universe_os")


@app.on_event("startup")
async def startup():
    """Initialize Universe OS on startup"""
    global universe_manager
    
    if not is_enabled("universe_os"):
        logger.warning("Universe OS is DISABLED via feature flag")
        return
    
    logger.info("🚀 Starting Universe OS...")
    universe_manager = UniverseManager()
    
    # Start background update task
    asyncio.create_task(update_universe_loop())
    logger.info("✅ Universe OS started successfully")


async def update_universe_loop():
    """Background task to update universe periodically"""
    while True:
        try:
            if universe_manager:
                universe_manager.update_universe()
                logger.info(f"Universe updated: {len(universe_manager.get_universe())} symbols")
        except Exception as e:
            health_checker.record_error(f"Universe update failed: {e}")
            logger.error(f"Universe update error: {e}")
        
        await asyncio.sleep(900)  # 15 minutes


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.get("/universe")
async def get_universe():
    """Get current trading universe"""
    if not universe_manager:
        return {"error": "Universe OS not initialized", "symbols": []}
    
    universe = universe_manager.get_universe()
    return {
        "symbols": universe,
        "count": len(universe),
        "timestamp": universe_manager.last_update.isoformat() if hasattr(universe_manager, 'last_update') else None
    }


@app.get("/symbol/{symbol}")
async def get_symbol_info(symbol: str):
    """Get information about a specific symbol"""
    if not universe_manager:
        return {"error": "Universe OS not initialized"}
    
    universe = universe_manager.get_universe()
    
    if symbol in universe:
        return {
            "symbol": symbol,
            "in_universe": True,
            "category": universe_manager.get_symbol_category(symbol) if hasattr(universe_manager, 'get_symbol_category') else "unknown"
        }
    else:
        return {
            "symbol": symbol,
            "in_universe": False
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)

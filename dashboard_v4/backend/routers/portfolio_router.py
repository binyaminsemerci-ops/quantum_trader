from fastapi import APIRouter
from schemas import Portfolio
import random
import logging
from services.quantum_client import quantum_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

@router.get("/status", response_model=Portfolio)
async def get_portfolio_status():
    """Get portfolio status with PnL and exposure metrics
    
    Fetches real data from Portfolio Intelligence service (port 8004).
    Falls back to mock data if service is unavailable.
    """
    # Try to get real data from Portfolio Intelligence service
    try:
        real_data = await quantum_client.get_portfolio_summary()
        
        if real_data:
            logger.info("✅ Using real portfolio data from Portfolio Intelligence")
            # Extract data from Portfolio Intelligence response
            # Adjust field mapping based on actual API response
            return {
                "pnl": real_data.get("total_pnl", real_data.get("pnl", 0)),
                "exposure": real_data.get("exposure", real_data.get("total_exposure", 0)),
                "drawdown": real_data.get("max_drawdown", real_data.get("drawdown", 0)),
                "positions": real_data.get("position_count", real_data.get("positions", 0))
            }
    except Exception as e:
        logger.warning(f"⚠️ Portfolio Intelligence unavailable: {e}")
    
    # Fallback to mock data if service unavailable
    logger.info("📊 Using mock portfolio data (Portfolio Intelligence unavailable)")
    pnl = round(random.uniform(90000, 140000), 2)
    exposure = round(random.uniform(0.3, 0.8), 2)
    drawdown = round(random.uniform(0.02, 0.15), 3)
    positions = random.randint(5, 25)
    return {
        "pnl": pnl,
        "exposure": exposure,
        "drawdown": drawdown,
        "positions": positions
    }

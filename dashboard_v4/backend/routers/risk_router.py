from fastapi import APIRouter
from schemas import Risk
import numpy as np
import random
import logging
from services.quantum_client import quantum_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/risk", tags=["Risk"])

@router.get("/metrics", response_model=Risk)
async def get_risk_metrics():
    """Get risk metrics including VaR, CVaR, and market regime
    
    Fetches real data from Risk Brain service (port 8012).
    Falls back to mock data if service is unavailable.
    """
    # Try to get real data from Risk Brain service
    try:
        real_data = await quantum_client.get_risk_metrics()
        
        if real_data:
            logger.info("✅ Using real risk data from Risk Brain")
            # Extract data from Risk Brain response
            # Adjust field mapping based on actual API response
            return {
                "var": real_data.get("var", real_data.get("value_at_risk", 0)),
                "cvar": real_data.get("cvar", real_data.get("conditional_var", 0)),
                "volatility": real_data.get("volatility", real_data.get("vol", 0)),
                "regime": real_data.get("market_regime", real_data.get("regime", "Unknown"))
            }
    except Exception as e:
        logger.warning(f"⚠️ Risk Brain unavailable: {e}")
    
    # Fallback to mock data if service unavailable
    logger.info("📊 Using mock risk data (Risk Brain unavailable)")
    # Simulate portfolio returns
    pnl_series = np.random.normal(0.001, 0.02, 1000)
    var_95 = np.percentile(pnl_series, 5)
    cvar_95 = pnl_series[pnl_series <= var_95].mean()
    volatility = float(np.std(pnl_series))
    regime = random.choice(["Bullish", "Bearish", "Neutral"])
    return {
        "var": round(var_95, 5),
        "cvar": round(cvar_95, 5),
        "volatility": round(volatility, 4),
        "regime": regime
    }

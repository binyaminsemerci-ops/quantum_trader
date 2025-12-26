from fastapi import APIRouter
from schemas import Risk
import numpy as np
import random

router = APIRouter(prefix="/risk", tags=["Risk"])

@router.get("/metrics", response_model=Risk)
def get_risk_metrics():
    """Get risk metrics including VaR, CVaR, and market regime"""
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

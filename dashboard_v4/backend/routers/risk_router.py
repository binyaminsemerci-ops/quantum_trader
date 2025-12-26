from fastapi import APIRouter
from schemas import Risk

router = APIRouter(prefix="/risk", tags=["Risk"])

@router.get("/metrics", response_model=Risk)
def get_risk_metrics():
    """Get risk metrics including VaR, CVaR, and market regime"""
    return Risk(
        var=-0.025,
        cvar=-0.034,
        volatility=0.18,
        regime="Bullish"
    )

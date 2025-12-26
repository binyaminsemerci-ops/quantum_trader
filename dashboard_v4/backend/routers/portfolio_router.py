from fastapi import APIRouter
from schemas import Portfolio

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

@router.get("/status", response_model=Portfolio)
def get_portfolio_status():
    """Get portfolio status with PnL and exposure metrics"""
    return Portfolio(
        pnl=125000.45,
        exposure=0.62,
        drawdown=0.08,
        positions=14
    )

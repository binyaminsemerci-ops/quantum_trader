from fastapi import APIRouter
from schemas import Portfolio
import random

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

@router.get("/status", response_model=Portfolio)
def get_portfolio_status():
    """Get portfolio status with PnL and exposure metrics"""
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

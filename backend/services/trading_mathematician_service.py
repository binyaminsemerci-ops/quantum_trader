"""
Trading Mathematician Service
Mathematical calculations for optimal trading parameters
"""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.services.ai.trading_mathematician import TradingMathematician
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Mathematician", version="1.0.0")

# Initialize
math_ai = None
health_checker = HealthChecker("trading_mathematician")


class ParameterRequest(BaseModel):
    """Request for optimal parameter calculation"""
    balance: float
    atr_pct: float
    win_rate: float
    symbol: str = "BTCUSDT"


@app.on_event("startup")
async def startup():
    """Initialize Trading Mathematician on startup"""
    global math_ai
    
    if not is_enabled("trading_mathematician"):
        logger.warning("Trading Mathematician is DISABLED via feature flag")
        return
    
    logger.info("🚀 Starting Trading Mathematician...")
    math_ai = TradingMathematician()
    logger.info("✅ Trading Mathematician started successfully")


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.post("/calculate_parameters")
async def calculate_parameters(request: ParameterRequest):
    """Calculate optimal trading parameters"""
    if not math_ai:
        raise HTTPException(status_code=503, detail="Trading Mathematician not initialized")
    
    try:
        params = math_ai.calculate_optimal_parameters(
            balance=request.balance,
            atr_pct=request.atr_pct,
            win_rate=request.win_rate,
            symbol=request.symbol
        )
        
        return {
            "symbol": request.symbol,
            "parameters": params,
            "status": "success"
        }
    except Exception as e:
        health_checker.record_error(f"Parameter calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/formulas")
async def get_formulas():
    """Get mathematical formulas used"""
    return {
        "kelly_criterion": "f = (bp - q) / b",
        "position_size": "size = (balance * risk_pct) / sl_pct",
        "risk_reward": "RR = tp_pct / sl_pct",
        "breakeven_wr": "BE_WR = 1 / (1 + RR)",
        "expected_value": "EV = (WR * avg_win) - ((1 - WR) * avg_loss)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)

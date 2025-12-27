"""
Portfolio Balancer AI (PBA) Service
Global portfolio state management
"""
import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.services.portfolio_balancer import PortfolioBalancer, Position, CandidateTrade
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled, get_mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Balancer AI", version="1.0.0")

# Initialize
pba = None
health_checker = HealthChecker("pba")


class PositionInput(BaseModel):
    """Input model for positions"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    margin: float
    leverage: float
    category: str = "EXPANSION"
    sector: str = "unknown"


class TradeInput(BaseModel):
    """Input model for candidate trades"""
    symbol: str
    action: str
    confidence: float
    size: float = 0.0
    margin_required: float = 0.0


@app.on_event("startup")
async def startup():
    """Initialize PBA on startup"""
    global pba
    
    if not is_enabled("pba"):
        logger.warning("PBA is DISABLED via feature flag")
        return
    
    mode = get_mode("pba")
    logger.info(f"🚀 Starting Portfolio Balancer AI (mode: {mode.value})...")
    pba = PortfolioBalancer()
    
    # Start background rebalancing
    asyncio.create_task(rebalancing_loop())
    logger.info(f"✅ PBA started successfully in {mode.value} mode")


async def rebalancing_loop():
    """Background task for periodic rebalancing"""
    while True:
        try:
            if pba:
                mode = get_mode("pba")
                if mode.value != "DISABLED":
                    # Run rebalancing logic
                    logger.info("Running portfolio rebalancing check...")
                    # pba.rebalance() - implement if needed
        except Exception as e:
            health_checker.record_error(f"Rebalancing failed: {e}")
            logger.error(f"Rebalancing error: {e}")
        
        await asyncio.sleep(300)  # 5 minutes


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.post("/evaluate_trade")
async def evaluate_trade(trade: TradeInput, positions: List[PositionInput]):
    """Evaluate if a trade should be allowed"""
    if not pba:
        raise HTTPException(status_code=503, detail="PBA not initialized")
    
    try:
        mode = get_mode("pba")
        
        # Convert inputs
        position_objs = [Position(**p.dict()) for p in positions]
        trade_obj = CandidateTrade(**trade.dict())
        
        # Evaluate
        decision = pba.evaluate_trade(trade_obj, position_objs)
        
        return {
            "allowed": decision["allowed"],
            "reason": decision.get("reason", ""),
            "adjustments": decision.get("adjustments", {}),
            "mode": mode.value
        }
    except Exception as e:
        health_checker.record_error(f"Trade evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio_state")
async def get_portfolio_state():
    """Get current portfolio state"""
    if not pba:
        raise HTTPException(status_code=503, detail="PBA not initialized")
    
    try:
        state = pba.get_portfolio_state()
        return state
    except Exception as e:
        health_checker.record_error(f"Portfolio state retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_mode/{mode}")
async def set_mode(mode: str):
    """Change PBA operating mode"""
    if mode.upper() not in ["OBSERVE", "ENFORCE"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use OBSERVE or ENFORCE")
    
    import os
    os.environ["PBA_MODE"] = mode.upper()
    
    logger.info(f"PBA mode changed to: {mode.upper()}")
    return {"status": "success", "mode": mode.upper()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

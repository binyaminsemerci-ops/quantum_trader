"""
Profit Amplification Layer (PAL) Service
Amplifies profit on winning trades
"""
import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.services.profit_amplification import ProfitAmplificationLayer, PositionSnapshot
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled, get_mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Profit Amplification Layer", version="1.0.0")

# Initialize
pal = None
health_checker = HealthChecker("pal")


class PositionInput(BaseModel):
    """Input model for position amplification evaluation"""
    symbol: str
    side: str
    current_R: float
    peak_R: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    drawdown_from_peak_R: float
    current_leverage: float
    position_size_usd: float
    risk_pct: float
    hold_time_hours: float
    pil_classification: str


@app.on_event("startup")
async def startup():
    """Initialize PAL on startup"""
    global pal
    
    if not is_enabled("pal"):
        logger.warning("PAL is DISABLED via feature flag")
        return
    
    mode = get_mode("pal")
    logger.info(f"🚀 Starting Profit Amplification Layer (mode: {mode.value})...")
    pal = ProfitAmplificationLayer()
    
    # Start background optimization
    asyncio.create_task(amplification_loop())
    logger.info(f"✅ PAL started successfully in {mode.value} mode")


async def amplification_loop():
    """Background task for profit amplification"""
    while True:
        try:
            if pal:
                mode = get_mode("pal")
                if mode.value != "DISABLED":
                    logger.debug("Running amplification analysis...")
                    # Amplification runs on-demand, not periodically
        except Exception as e:
            health_checker.record_error(f"Amplification failed: {e}")
            logger.error(f"Amplification error: {e}")
        
        await asyncio.sleep(300)  # 5 minutes


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.post("/evaluate_amplification")
async def evaluate_amplification(position: PositionInput):
    """Evaluate if position should be amplified"""
    if not pal:
        raise HTTPException(status_code=503, detail="PAL not initialized")
    
    try:
        mode = get_mode("pal")
        
        # Convert to PositionSnapshot
        snapshot = PositionSnapshot(**position.dict())
        
        # Evaluate
        decision = pal.evaluate_position(snapshot)
        
        return {
            "symbol": position.symbol,
            "action": decision["action"],
            "rationale": decision.get("rationale", ""),
            "parameters": decision.get("parameters", {}),
            "mode": mode.value
        }
    except Exception as e:
        health_checker.record_error(f"Amplification evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get PAL performance statistics"""
    if not pal:
        raise HTTPException(status_code=503, detail="PAL not initialized")
    
    try:
        stats = pal.get_statistics()
        return stats
    except Exception as e:
        health_checker.record_error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_mode/{mode}")
async def set_mode(mode: str):
    """Change PAL operating mode"""
    if mode.upper() not in ["OBSERVE", "HEDGEFUND", "AGGRESSIVE"]:
        raise HTTPException(status_code=400, detail="Invalid mode")
    
    import os
    os.environ["PAL_MODE"] = mode.upper()
    
    logger.info(f"PAL mode changed to: {mode.upper()}")
    return {"status": "success", "mode": mode.upper()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

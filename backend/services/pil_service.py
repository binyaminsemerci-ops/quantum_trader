"""
Position Intelligence Layer Service
Classifies positions and provides intelligence
"""
import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.services.position_intelligence import PositionIntelligenceLayer
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Position Intelligence Layer", version="1.0.0")

# Initialize
pil = None
health_checker = HealthChecker("pil")


class PositionData(BaseModel):
    """Position data for classification"""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    hold_time_hours: float


@app.on_event("startup")
async def startup():
    """Initialize PIL on startup"""
    global pil
    
    if not is_enabled("pil"):
        logger.warning("PIL is DISABLED via feature flag")
        return
    
    logger.info("🚀 Starting Position Intelligence Layer...")
    pil = PositionIntelligenceLayer()
    logger.info("✅ PIL started successfully")


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.post("/classify")
async def classify_position(position: PositionData):
    """Classify a position"""
    if not pil:
        raise HTTPException(status_code=503, detail="PIL not initialized")
    
    try:
        # Convert to dict for PIL
        position_dict = position.dict()
        
        # Classify
        classification = pil.classify_position(position_dict)
        
        return {
            "symbol": position.symbol,
            "category": classification.category.value,
            "recommendation": classification.recommendation.value,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
            "metrics": classification.metrics
        }
    except Exception as e:
        health_checker.record_error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classifications")
async def get_all_classifications():
    """Get all current position classifications"""
    if not pil:
        raise HTTPException(status_code=503, detail="PIL not initialized")
    
    return {
        "classifications": [
            {
                "symbol": c.symbol,
                "category": c.category.value,
                "recommendation": c.recommendation.value,
                "confidence": c.confidence
            }
            for c in pil.classifications.values()
        ],
        "count": len(pil.classifications)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)

"""Strategy Brain Service - Evaluates trading signals for strategy approval.

Phase 2.2: Stub implementation - returns default approval
Future: Implement full strategy evaluation logic
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategy Brain Service", version="1.0.0")


class SignalRequest(BaseModel):
    """Request model for signal evaluation."""
    symbol: str
    direction: str  # BUY/SELL
    confidence: float
    entry_price: Optional[float] = None
    regime: Optional[str] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "strategy_brain",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0 (stub)",
    }


@app.post("/evaluate")
async def evaluate(request: SignalRequest):
    """Evaluate trading signal.
    
    Phase 2.2: Stub implementation - always approves
    Future: Implement strategy filters, regime checks, etc.
    
    Args:
        request: Signal to evaluate
    
    Returns:
        Approval decision with reasoning
    """
    logger.info(f"📊 Strategy evaluation: {request.symbol} {request.direction} (confidence={request.confidence:.2f})")
    
    # Phase 2.2: Always approve for now
    # Future logic:
    # - Check if signal aligns with current strategy
    # - Verify market regime compatibility
    # - Check for overtrading
    # - Validate symbol eligibility
    
    return {
        "approved": True,
        "reason": "Strategy Brain stub - default approve",
        "confidence": 0.8,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("📊 Starting Strategy Brain Service on port 8011...")
    uvicorn.run(app, host="0.0.0.0", port=8011)

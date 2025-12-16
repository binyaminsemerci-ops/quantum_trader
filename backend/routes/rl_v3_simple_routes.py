"""
RL v3 Prediction API Routes
============================

Simple REST API endpoint for RL v3 (PPO) predictions.

Endpoint:
- POST /api/v1/rl-v3/predict - Get PPO prediction with minimal schema

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any
import structlog

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/rl-v3", tags=["RL v3 Predictions"])


class RLv3PredictRequest(BaseModel):
    """Simplified request model for RL v3 prediction."""
    
    price_change_1m: float = Field(default=0.0, description="1-minute price change")
    price_change_5m: float = Field(default=0.0, description="5-minute price change")
    price_change_15m: float = Field(default=0.0, description="15-minute price change")
    volatility: float = Field(default=0.02, description="Market volatility")
    rsi: float = Field(default=50.0, description="RSI indicator")
    macd: float = Field(default=0.0, description="MACD indicator")
    position_size: float = Field(default=0.0, description="Current position size")
    position_side: float = Field(default=0.0, description="Position side (1=long, -1=short, 0=none)")
    balance: float = Field(default=10000.0, description="Account balance")
    equity: float = Field(default=10000.0, description="Account equity")
    regime: str = Field(default="TREND", description="Market regime")
    trend_strength: float = Field(default=0.5, description="Trend strength (0-1)")
    volume_ratio: float = Field(default=1.0, description="Volume ratio")
    bid_ask_spread: float = Field(default=0.001, description="Bid-ask spread")
    time_of_day: float = Field(default=0.5, description="Time of day (0-1)")
    
    class Config:
        extra = "allow"  # Allow extra fields


class RLv3PredictResponse(BaseModel):
    """Response model for RL v3 prediction."""
    
    action: int = Field(..., description="Action code (0-5)")
    confidence: float = Field(..., description="Confidence score (0-1)")


@router.post("/predict", response_model=RLv3PredictResponse)
async def rl_v3_predict(
    request: Request,
    body: RLv3PredictRequest
):
    """
    Get RL v3 (PPO) prediction for given observation.
    
    Action codes:
    - 0: HOLD
    - 1: LONG
    - 2: SHORT
    - 3: REDUCE
    - 4: CLOSE
    - 5: EMERGENCY_FLATTEN
    
    Args:
        request: FastAPI request (provides access to app.state)
        body: Observation data
        
    Returns:
        Action code and confidence score
        
    Raises:
        HTTPException 503: RL v3 manager not initialized
        HTTPException 500: Prediction failed
    """
    try:
        # Get RL v3 manager from app state
        rl_manager = getattr(request.app.state, 'rl_v3_manager', None)
        
        if rl_manager is None:
            logger.error("[RL v3 API] Manager not initialized")
            raise HTTPException(
                status_code=503,
                detail="RL v3 manager not initialized. Check server logs."
            )
        
        # Convert request to observation dict
        obs_dict = body.dict()
        
        logger.debug(
            "[RL v3 API] Prediction request",
            regime=obs_dict.get("regime"),
            rsi=obs_dict.get("rsi"),
            volatility=obs_dict.get("volatility")
        )
        
        # Get prediction from RL manager
        result = rl_manager.predict(obs_dict)
        
        logger.info(
            "[RL v3 API] Prediction success",
            action=result['action'],
            confidence=result['confidence']
        )
        
        return RLv3PredictResponse(
            action=result['action'],
            confidence=result['confidence']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "[RL v3 API] Prediction failed",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

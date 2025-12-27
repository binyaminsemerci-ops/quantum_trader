"""
Risk & Safety Service - Simplified Stub Version

Lightweight API endpoint for risk validation.
Returns permissive responses for testnet trading.
"""
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk & Safety Service (Stub)",
    description="Simplified risk validation for testnet trading",
    version="1.0.0-stub"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskValidationRequest(BaseModel):
    """Risk validation request."""
    symbol: str
    side: str  # BUY/SELL/LONG/SHORT
    size: float  # Position size in USD
    leverage: int = 1
    account_balance: Optional[float] = None


class RiskValidationResponse(BaseModel):
    """Risk validation response."""
    allowed: bool
    reason: Optional[str] = None
    max_size_usd: Optional[float] = None
    max_leverage: Optional[int] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "service": "risk-safety-stub",
        "status": "OK",
        "version": "1.0.0-stub",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "PERMISSIVE",
        "note": "Stub implementation for testnet - all trades allowed"
    }


@app.post("/validate", response_model=RiskValidationResponse)
async def validate_trade(request: RiskValidationRequest):
    """
    Validate a trade request.
    
    Stub implementation: Always allows trades for testnet.
    """
    logger.info(
        f"[RISK-STUB] Validating: {request.symbol} {request.side} "
        f"${request.size} @ {request.leverage}x"
    )
    
    # In testnet mode, allow everything
    return RiskValidationResponse(
        allowed=True,
        reason=None,
        max_size_usd=10000.0,  # $10k max per trade
        max_leverage=30  # 30x max leverage
    )


@app.get("/ess/status")
async def get_ess_status():
    """Get Emergency Stop System status."""
    return {
        "state": "ARMED",
        "enabled": True,
        "can_execute_orders": True,
        "metrics": {
            "daily_drawdown_pct": 0.0,
            "open_loss_pct": 0.0,
            "execution_errors": 0
        },
        "note": "Stub implementation - ESS always ARMED"
    }


@app.get("/policy")
async def get_policy():
    """Get current policy settings."""
    return {
        "ess": {
            "enabled": True,
            "max_daily_dd_pct": 15.0,
            "max_open_loss_pct": 25.0,
            "max_execution_errors": 10
        },
        "risk": {
            "max_position_usd": 2000.0,
            "min_position_usd": 100.0,
            "max_leverage": 30,
            "max_exposure_pct": 200.0
        },
        "note": "Stub implementation - permissive limits for testnet"
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", "8005"))
    
    logger.info(f"[RISK-STUB] Starting Risk & Safety Service (Stub) on port {port}")
    logger.info("[RISK-STUB] Mode: PERMISSIVE (testnet)")
    logger.info("[RISK-STUB] All trades will be allowed")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

"""CEO Brain FastAPI Service - HTTP interface for CEO Brain."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import os

from backend.ai_orchestrator.ceo_brain import CEOBrain, SystemState, OperatingMode
from backend.ai_orchestrator.ceo_policy import MarketRegime, SystemHealth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CEO Brain Service", version="1.0.0")

# Global CEO Brain instance
_ceo_brain: Optional[CEOBrain] = None


def get_ceo_brain() -> CEOBrain:
    """Get or create CEO Brain singleton."""
    global _ceo_brain
    if _ceo_brain is None:
        _ceo_brain = CEOBrain()
        logger.info("🧠 CEO Brain initialized")
    return _ceo_brain


class SystemStateRequest(BaseModel):
    """Request model for system state."""
    risk_score: Optional[float] = 50.0
    win_rate: Optional[float] = 0.5
    sharpe_ratio: Optional[float] = 0.0
    current_drawdown: Optional[float] = 0.0
    max_drawdown_today: Optional[float] = 0.0
    open_positions: Optional[int] = 0
    market_regime: Optional[str] = "TRENDING"
    system_health: Optional[str] = "HEALTHY"


@app.get("/health")
async def health():
    """Health check endpoint."""
    ceo_brain = get_ceo_brain()
    
    return {
        "status": "healthy",
        "service": "ceo_brain",
        "timestamp": datetime.utcnow().isoformat(),
        "enabled": os.getenv("ENABLE_CEO_BRAIN", "true") == "true",
        "current_mode": ceo_brain._current_mode.value,
        "decisions_made": len(ceo_brain._decision_history),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "CEO Brain",
        "description": "AI CEO orchestration and decision-making service",
        "version": "1.0.0",
        "endpoints": ["/health", "/status", "/decide", "/mode"],
    }


@app.get("/status")
async def status():
    """Get current CEO status with operating mode."""
    ceo_brain = get_ceo_brain()
    
    return {
        "status": "active",
        "operating_mode": ceo_brain._current_mode.value,
        "confidence": 0.85,  # Default confidence when no recent decision
        "mode": ceo_brain._current_mode.value,  # Legacy field
        "timestamp": datetime.utcnow().isoformat(),
        "last_transition": datetime.fromtimestamp(ceo_brain._last_transition_time).isoformat(),
        "decisions_in_history": len(ceo_brain._decision_history),
    }


@app.get("/mode")
async def get_mode():
    """Get current operating mode (simple endpoint for orchestrator)."""
    ceo_brain = get_ceo_brain()
    
    return {
        "mode": ceo_brain._current_mode.value,
        "confidence": 0.85,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/decide")
async def decide(request: SystemStateRequest):
    """Make CEO decision based on system state.
    
    Args:
        request: System state with risk, performance, and market metrics
    
    Returns:
        CEO decision with operating mode, reasoning, and policy updates
    """
    logger.info("🧠 CEO decision requested")
    
    try:
        ceo_brain = get_ceo_brain()
        
        # Convert request to SystemState
        state = SystemState(
            # Risk metrics
            risk_score=request.risk_score or 50.0,
            var_estimate=0.0,  # Not provided yet
            expected_shortfall=0.0,  # Not provided yet
            tail_risk_score=0.0,  # Not provided yet
            
            # Performance metrics
            win_rate=request.win_rate or 0.5,
            sharpe_ratio=request.sharpe_ratio or 0.0,
            profit_factor=1.0,  # Not provided yet
            avg_win_loss_ratio=1.0,  # Not provided yet
            
            # Portfolio metrics
            current_drawdown=request.current_drawdown or 0.0,
            max_drawdown_today=request.max_drawdown_today or 0.0,
            total_pnl_today=0.0,  # Not provided yet
            open_positions=request.open_positions or 0,
            total_exposure=0.0,  # Not provided yet
            
            # Market state
            market_regime=MarketRegime(request.market_regime or "TRENDING"),
            regime_confidence=0.8,  # Default
            market_volatility=0.5,  # Default
            
            # System health
            system_health=SystemHealth(request.system_health or "HEALTHY"),
            ai_engine_healthy=True,
            rl_engine_healthy=True,
            execution_healthy=True,
            risk_os_healthy=True,
            
            # Timing
            timestamp=datetime.utcnow(),
        )
        
        # Get CEO decision
        decision = ceo_brain.evaluate(state, current_mode=ceo_brain._current_mode)
        
        logger.info(
            f"🧠 CEO Decision: {decision.operating_mode.value} "
            f"(changed={decision.mode_changed}, confidence={decision.decision_confidence:.2f})"
        )
        
        return {
            "operating_mode": decision.operating_mode.value,
            "mode_changed": decision.mode_changed,
            "primary_reason": decision.primary_reason,
            "contributing_factors": decision.contributing_factors,
            "decision_confidence": decision.decision_confidence,
            "alert_level": decision.alert_level,
            "alert_message": decision.alert_message,
            "update_policy_store": decision.update_policy_store,
            "policy_updates": decision.policy_updates,
            "timestamp": decision.decision_timestamp.isoformat(),
        }
    
    except Exception as e:
        logger.error(f"🧠 CEO decision failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("🧠 Starting CEO Brain Service on port 8010...")
    uvicorn.run(app, host="0.0.0.0", port=8010)


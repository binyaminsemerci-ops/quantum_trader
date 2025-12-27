"""
Risk v3 FastAPI Application

EPIC-RISK3-001: REST API for Risk v3 service

Endpoints:
- GET /risk/health - Health check
- GET /risk/status - Orchestrator status
- GET /risk/snapshot - Current risk snapshot
- GET /risk/exposure - Exposure matrix
- GET /risk/var - VaR/ES results
- GET /risk/systemic - Systemic risk signals
- POST /risk/evaluate - Trigger risk evaluation
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from .orchestrator import RiskOrchestrator
from .models import GlobalRiskSignal

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Risk v3 API",
    description="Global Risk Engine v3 for Quantum Trader",
    version="3.0.0",
)

# Global orchestrator instance (initialized in main.py)
orchestrator: Optional[RiskOrchestrator] = None


def init_orchestrator(orch: RiskOrchestrator):
    """Initialize orchestrator (called from main.py)"""
    global orchestrator
    orchestrator = orch
    logger.info("[RISK-V3-API] Orchestrator initialized")


@app.get("/risk/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "risk_v3",
        "version": "3.0.0",
        "orchestrator_ready": orchestrator is not None,
    }


@app.get("/risk/status")
async def get_status():
    """Get orchestrator status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = await orchestrator.get_status()
    return JSONResponse(content=status)


@app.get("/risk/snapshot")
async def get_snapshot():
    """Get current risk snapshot"""
    if not orchestrator or not orchestrator.last_evaluation:
        raise HTTPException(status_code=404, detail="No risk evaluation available yet")
    
    snapshot = orchestrator.last_evaluation.snapshot
    return JSONResponse(content=snapshot.model_dump(mode="json"))


@app.get("/risk/exposure")
async def get_exposure():
    """Get exposure matrix"""
    if not orchestrator or not orchestrator.last_evaluation:
        raise HTTPException(status_code=404, detail="No risk evaluation available yet")
    
    exposure_matrix = orchestrator.last_evaluation.exposure_matrix
    return JSONResponse(content=exposure_matrix.model_dump(mode="json"))


@app.get("/risk/var")
async def get_var():
    """Get VaR/ES results"""
    if not orchestrator or not orchestrator.last_evaluation:
        raise HTTPException(status_code=404, detail="No risk evaluation available yet")
    
    var_result = orchestrator.last_evaluation.var_result
    es_result = orchestrator.last_evaluation.es_result
    
    if not var_result or not es_result:
        raise HTTPException(status_code=404, detail="VaR/ES not calculated")
    
    return JSONResponse(content={
        "var": var_result.model_dump(mode="json"),
        "es": es_result.model_dump(mode="json"),
    })


@app.get("/risk/systemic")
async def get_systemic():
    """Get systemic risk signals"""
    if not orchestrator or not orchestrator.last_evaluation:
        raise HTTPException(status_code=404, detail="No risk evaluation available yet")
    
    systemic_signals = orchestrator.last_evaluation.systemic_signals
    return JSONResponse(content={
        "count": len(systemic_signals),
        "signals": [s.model_dump(mode="json") for s in systemic_signals],
    })


@app.post("/risk/evaluate")
async def trigger_evaluation(force_refresh: bool = False):
    """
    Trigger manual risk evaluation
    
    Args:
        force_refresh: Force refresh of all data sources
    
    Returns:
        GlobalRiskSignal
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        signal = await orchestrator.evaluate_risk(force_refresh=force_refresh)
        return JSONResponse(content=signal.model_dump(mode="json"))
    except Exception as e:
        logger.error(f"[RISK-V3-API] Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Risk evaluation failed: {str(e)}")


@app.get("/risk/global")
async def get_global_signal():
    """Get complete global risk signal"""
    if not orchestrator or not orchestrator.last_evaluation:
        raise HTTPException(status_code=404, detail="No risk evaluation available yet")
    
    return JSONResponse(content=orchestrator.last_evaluation.model_dump(mode="json"))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "detail": str(exc.detail) if hasattr(exc, "detail") else str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"[RISK-V3-API] Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

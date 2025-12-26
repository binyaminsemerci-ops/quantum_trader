"""
Integration Router - Direct access to all Quantum Trader services
"""
from fastapi import APIRouter, HTTPException
from services.quantum_client import quantum_client
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/integrations", tags=["Integrations"])

@router.get("/portfolio/summary")
async def get_portfolio_summary():
    """Get portfolio summary from Portfolio Intelligence"""
    data = await quantum_client.get_portfolio_summary()
    if data is None:
        raise HTTPException(status_code=503, detail="Portfolio Intelligence service unavailable")
    return data

@router.get("/portfolio/positions")
async def get_portfolio_positions():
    """Get current positions"""
    data = await quantum_client.get_portfolio_positions()
    if data is None:
        raise HTTPException(status_code=503, detail="Portfolio Intelligence service unavailable")
    return data

@router.get("/trades/active")
async def get_active_trades():
    """Get active trades from Trading Bot"""
    data = await quantum_client.get_live_trades()
    if data is None:
        raise HTTPException(status_code=503, detail="Trading Bot service unavailable")
    return data

@router.get("/trades/history")
async def get_trade_history(limit: int = 100):
    """Get trade history"""
    data = await quantum_client.get_trade_history(limit)
    if data is None:
        raise HTTPException(status_code=503, detail="Trading Bot service unavailable")
    return data

@router.get("/ai/predictions")
async def get_ai_predictions():
    """Get AI model predictions"""
    data = await quantum_client.get_ai_predictions()
    if data is None:
        raise HTTPException(status_code=503, detail="AI Engine service unavailable")
    return data

@router.get("/ai/models")
async def get_model_performance():
    """Get model performance metrics"""
    data = await quantum_client.get_model_performance()
    if data is None:
        raise HTTPException(status_code=503, detail="AI Engine service unavailable")
    return data

@router.get("/risk/var")
async def get_risk_var():
    """Get Value at Risk"""
    data = await quantum_client.get_risk_var()
    if data is None:
        raise HTTPException(status_code=503, detail="Risk Brain service unavailable")
    return data

@router.get("/risk/exposure")
async def get_risk_exposure():
    """Get current risk exposure"""
    data = await quantum_client.get_risk_exposure()
    if data is None:
        raise HTTPException(status_code=503, detail="Risk Brain service unavailable")
    return data

@router.get("/strategy/performance")
async def get_strategy_performance():
    """Get strategy performance"""
    data = await quantum_client.get_strategy_performance()
    if data is None:
        raise HTTPException(status_code=503, detail="Strategy Brain service unavailable")
    return data

@router.get("/health/all")
async def check_all_services():
    """Check health of all Quantum services"""
    services = ['portfolio', 'trading', 'ai_engine', 'risk', 'strategy', 'ceo', 
                'model_supervisor', 'universe', 'backend']
    
    health_status = {}
    for service in services:
        is_healthy = await quantum_client.health_check(service)
        health_status[service] = {
            "status": "healthy" if is_healthy else "unavailable",
            "url": quantum_client.SERVICES.get(service)
        }
    
    return health_status

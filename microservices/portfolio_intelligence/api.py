"""
Portfolio Intelligence Service - API Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from microservices.portfolio_intelligence.service import PortfolioIntelligenceService
from microservices.portfolio_intelligence.models import (
    PortfolioSnapshot, PnLBreakdown, ExposureBreakdown, DrawdownMetrics, ServiceHealth
)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])
logger = logging.getLogger(__name__)

# Global service reference - set by main.py during startup
_service_instance: Optional[PortfolioIntelligenceService] = None


def set_service_instance(service: PortfolioIntelligenceService):
    """Set the global service instance (called by main.py during startup)."""
    global _service_instance
    _service_instance = service
    logger.info("[API] Service instance registered")


def get_service():
    """Dependency: Get service instance."""
    if _service_instance is None:
        logger.error("[API] Service instance is None!")
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _service_instance
    return service


@router.get("/health", response_model=ServiceHealth)
async def health_check(service: PortfolioIntelligenceService = Depends(get_service)):
    """
    Get service health status.
    
    Returns:
        ServiceHealth: Service health with component status
    """
    health = await service.get_health()
    return health


@router.get("/snapshot", response_model=PortfolioSnapshot)
async def get_snapshot(service: PortfolioIntelligenceService = Depends(get_service)):
    """
    Get current portfolio snapshot.
    
    Returns:
        PortfolioSnapshot: Complete portfolio state (equity, positions, PnL, etc.)
    """
    snapshot = service.get_current_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="No snapshot available")
    return snapshot


@router.get("/pnl", response_model=PnLBreakdown)
async def get_pnl(service: PortfolioIntelligenceService = Depends(get_service)):
    """
    Get PnL breakdown (realized, unrealized, daily, weekly, monthly).
    
    Returns:
        PnLBreakdown: Detailed PnL metrics
    """
    pnl = await service.get_pnl_breakdown()
    return pnl


@router.get("/exposure", response_model=ExposureBreakdown)
async def get_exposure(service: PortfolioIntelligenceService = Depends(get_service)):
    """
    Get exposure breakdown by symbol/sector.
    
    Returns:
        ExposureBreakdown: Total, long, short, net exposure
    """
    exposure = await service.get_exposure_breakdown()
    return exposure


@router.get("/drawdown", response_model=DrawdownMetrics)
async def get_drawdown(service: PortfolioIntelligenceService = Depends(get_service)):
    """
    Get drawdown metrics (daily, weekly, max).
    
    Returns:
        DrawdownMetrics: Drawdown analysis
    """
    drawdown = await service.get_drawdown_metrics()
    return drawdown

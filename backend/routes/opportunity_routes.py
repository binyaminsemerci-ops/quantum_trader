"""
FastAPI Integration Example for OpportunityRanker

This demonstrates how to integrate the OpportunityRanker into your
Quantum Trader FastAPI backend.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from backend.services.opportunity_ranker import OpportunityRanker

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(tags=["Opportunity Ranking"])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_opportunity_ranker(request: Request) -> Optional[OpportunityRanker]:
    """
    Retrieve the global OpportunityRanker instance from app state.
    
    Returns:
        OpportunityRanker instance or None if not initialized
    """
    return getattr(request.app.state, "opportunity_ranker", None)


# ============================================================================
# PYDANTIC MODELS (Response Schemas)
# ============================================================================

from pydantic import BaseModel, Field

class OpportunityRanking(BaseModel):
    """Response model for opportunity rankings."""
    symbol: str
    score: float = Field(..., ge=0.0, le=1.0, description="Opportunity score 0.0-1.0")
    rank: int = Field(..., ge=1, description="Ranking position (1 = best)")


class OpportunityRankingsResponse(BaseModel):
    """Response for GET /rankings endpoint."""
    rankings: List[OpportunityRanking]
    total_symbols: int
    last_updated: datetime
    update_interval_minutes: int = 15


class DetailedMetricsResponse(BaseModel):
    """Response for GET /rankings/{symbol}/details endpoint."""
    symbol: str
    trend_strength: float
    volatility_quality: float
    liquidity_score: float
    spread_score: float
    symbol_winrate_score: float
    regime_score: float
    noise_score: float
    final_score: float
    timestamp: datetime


class RefreshResponse(BaseModel):
    """Response for POST /rankings/refresh endpoint."""
    status: str
    symbols_ranked: int
    top_5: List[str]
    execution_time_seconds: float


class TopSymbolsResponse(BaseModel):
    """Response for GET /rankings/top endpoint."""
    symbols: List[OpportunityRanking]
    count: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/rankings", response_model=OpportunityRankingsResponse)
async def get_current_rankings(
    request: Request,
    min_score: float = 0.0
):
    """
    Get current opportunity rankings for all symbols.
    
    Args:
        min_score: Optional minimum score filter (0.0-1.0)
        
    Returns:
        Current rankings with scores and metadata
    """
    ranker = get_opportunity_ranker(request)
    if not ranker:
        raise HTTPException(
            status_code=503,
            detail="OpportunityRanker not initialized"
        )
    
    try:
        # Get all rankings
        rankings = ranker.get_rankings()
        
        if not rankings:
            raise HTTPException(
                status_code=404,
                detail="No rankings available. System is initializing or refresh needed."
            )
        
        # Filter by min_score
        filtered = [r for r in rankings if r.overall_score >= min_score]
        
        # Convert to response format
        ranking_list = [
            OpportunityRanking(
                symbol=r.symbol,
                score=r.overall_score,
                rank=r.rank
            )
            for r in filtered
        ]
        
        return OpportunityRankingsResponse(
            rankings=ranking_list,
            total_symbols=len(ranking_list),
            last_updated=filtered[0].timestamp if filtered else datetime.utcnow(),
            update_interval_minutes=5,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rankings/top", response_model=TopSymbolsResponse)
async def get_top_symbols(
    request: Request,
    n: int = 10,
    min_score: float = 0.5
):
    """
    Get top N symbols by opportunity score.
    
    Args:
        n: Number of top symbols to return (default: 10)
        min_score: Minimum score threshold (default: 0.5)
        
    Returns:
        List of top N symbols with scores
    """
    ranker = get_opportunity_ranker(request)
    if not ranker:
        raise HTTPException(
            status_code=503,
            detail="OpportunityRanker not initialized"
        )
    
    try:
        # Get top opportunities
        top_rankings = ranker.get_top_opportunities(n=n, min_score=min_score)
        
        if not top_rankings:
            raise HTTPException(
                status_code=404,
                detail="No symbols meet the criteria"
            )
        
        rankings = [
            OpportunityRanking(
                symbol=r.symbol,
                score=r.overall_score,
                rank=r.rank
            )
            for r in top_rankings
        ]
        
        return TopSymbolsResponse(
            symbols=rankings,
            count=len(rankings),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve top symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rankings/{symbol}", response_model=OpportunityRanking)
async def get_symbol_ranking(
    symbol: str,
    request: Request
):
    """
    Get opportunity score for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        
    Returns:
        Symbol's current ranking and score
    """
    ranker = get_opportunity_ranker(request)
    if not ranker:
        raise HTTPException(
            status_code=503,
            detail="OpportunityRanker not initialized"
        )
    
    try:
        ranking = ranker.get_ranking_for_symbol(symbol)
        
        if not ranking:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found in rankings"
            )
        
        return OpportunityRanking(
            symbol=ranking.symbol,
            score=ranking.overall_score,
            rank=ranking.rank,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve ranking for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rankings/{symbol}/details")
async def get_symbol_detailed_metrics(
    symbol: str,
    request: Request
):
    """
    Get detailed metric breakdown for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        
    Returns:
        Detailed metrics including all individual scores
    """
    ranker = get_opportunity_ranker(request)
    if not ranker:
        raise HTTPException(
            status_code=503,
            detail="OpportunityRanker not initialized"
        )
    
    try:
        ranking = ranker.get_ranking_for_symbol(symbol)
        
        if not ranking:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found in rankings"
            )
        
        return {
            "symbol": ranking.symbol,
            "overall_score": ranking.overall_score,
            "rank": ranking.rank,
            "metric_scores": ranking.metric_scores,
            "metadata": ranking.metadata,
            "timestamp": ranking.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compute detailed metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_rankings(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger a ranking update.
    
    This will compute fresh opportunity scores for all symbols and
    update the rankings store.
    
    Returns:
        Refresh status and execution details
    """
    ranker = get_opportunity_ranker(request)
    if not ranker:
        raise HTTPException(
            status_code=503,
            detail="OpportunityRanker not initialized"
        )
    
    try:
        start_time = datetime.utcnow()
        
        # Get symbols from factory
        from backend.integrations.opportunity_ranker_factory import get_default_symbols
        symbols = get_default_symbols()
        
        # Update rankings asynchronously
        await ranker.rank_opportunities(symbols)
        
        rankings = ranker.get_rankings()
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        top_5 = [r.symbol for r in rankings[:5]] if rankings else []
        
        logger.info(
            f"Rankings refreshed: {len(rankings)} symbols "
            f"(execution_time={execution_time:.2f}s)"
        )
        
        return {
            "status": "success",
            "message": f"Refreshed {len(rankings)} symbols",
            "ranking_count": len(rankings),
            "top_5": top_5,
            "execution_time_seconds": execution_time,
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BACKGROUND TASK (Periodic Updates)
# ============================================================================

async def periodic_ranking_updater(
    ranker: OpportunityRanker,
    interval_minutes: int = 15
):
    """
    Background task that periodically updates opportunity rankings.
    
    Args:
        ranker: OpportunityRanker instance
        interval_minutes: Update interval in minutes
    """
    logger.info(f"Starting periodic ranking updater (interval={interval_minutes}m)")
    
    while True:
        try:
            rankings = ranker.update_rankings()
            logger.info(
                f"Periodic update completed: {len(rankings)} symbols ranked"
            )
            
        except Exception as e:
            logger.error(f"Periodic ranking update failed: {e}")
        
        # Wait for next interval
        await asyncio.sleep(interval_minutes * 60)


# ============================================================================
# APP STARTUP/SHUTDOWN HOOKS
# ============================================================================

async def startup_opportunity_ranker(app):
    """
    Initialize OpportunityRanker on app startup.
    
    Add this to your FastAPI app startup:
    
    @app.on_event("startup")
    async def startup_event():
        await startup_opportunity_ranker(app)
    """
    from opportunity_ranker import OpportunityRanker
    
    # Initialize dependencies (these should come from your service layer)
    market_data = app.state.market_data_client
    trade_logs = app.state.trade_log_repository
    regime_detector = app.state.regime_detector
    opportunity_store = app.state.opportunity_store
    
    # Configuration
    config = app.state.config
    symbols = config.TRADEABLE_SYMBOLS
    
    # Create ranker
    ranker = OpportunityRanker(
        market_data=market_data,
        trade_logs=trade_logs,
        regime_detector=regime_detector,
        opportunity_store=opportunity_store,
        symbols=symbols,
        timeframe=config.OPPORTUNITY_RANKER_TIMEFRAME,
        candle_limit=config.OPPORTUNITY_RANKER_CANDLE_LIMIT,
        min_score_threshold=config.OPPORTUNITY_MIN_SCORE,
    )
    
    # Store in app state
    app.state.opportunity_ranker = ranker
    
    # Compute initial rankings
    logger.info("Computing initial opportunity rankings...")
    ranker.update_rankings()
    logger.info("Initial rankings computed successfully")
    
    # Start background updater
    update_interval = config.OPPORTUNITY_UPDATE_INTERVAL_MINUTES
    asyncio.create_task(periodic_ranking_updater(ranker, update_interval))
    
    logger.info("OpportunityRanker initialized and background updater started")


# ============================================================================
# EXAMPLE USAGE IN ANOTHER SERVICE
# ============================================================================

class OrchestratorWithOpportunityFilter:
    """
    Example showing how other services can use OpportunityRanker.
    """
    
    def __init__(self, opportunity_store, min_opportunity_score: float = 0.5):
        self.opportunity_store = opportunity_store
        self.min_opportunity_score = min_opportunity_score
    
    async def should_allow_trade(self, signal) -> tuple[bool, str]:
        """
        Check if trade should be allowed based on opportunity score.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Get current rankings
        rankings = self.opportunity_store.get()
        
        symbol_score = rankings.get(signal.symbol, 0.0)
        
        if symbol_score < self.min_opportunity_score:
            return False, f"Symbol opportunity score too low ({symbol_score:.3f})"
        
        if signal.symbol not in rankings:
            return False, f"Symbol {signal.symbol} not in opportunity rankings"
        
        # Symbol passes opportunity filter
        return True, f"Opportunity score: {symbol_score:.3f}"


# ============================================================================
# CONFIGURATION EXAMPLE
# ============================================================================

class OpportunityRankerConfig:
    """Example configuration for OpportunityRanker."""
    
    # Symbols to track
    TRADEABLE_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT",
        "MATICUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT",
    ]
    
    # OpportunityRanker settings
    OPPORTUNITY_RANKER_TIMEFRAME = "1h"
    OPPORTUNITY_RANKER_CANDLE_LIMIT = 200
    OPPORTUNITY_MIN_SCORE = 0.5
    OPPORTUNITY_UPDATE_INTERVAL_MINUTES = 15
    
    # Custom weights (optional)
    OPPORTUNITY_WEIGHTS = {
        'trend_strength': 0.25,
        'volatility_quality': 0.20,
        'liquidity_score': 0.15,
        'regime_score': 0.15,
        'symbol_winrate_score': 0.10,
        'spread_score': 0.10,
        'noise_score': 0.05,
    }

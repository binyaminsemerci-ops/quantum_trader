"""
Analytics API - Fund Manager Oversight Dashboard

Provides comprehensive analytics endpoints for monitoring system performance,
strategy attribution, model comparison, and risk metrics.

Endpoints:
- GET /api/analytics/daily - Daily performance summary
- GET /api/analytics/strategies - Strategy attribution analysis
- GET /api/analytics/models - Model performance comparison
- GET /api/analytics/risk - Risk metrics dashboard
- GET /api/analytics/opportunities - Opportunity trends over time
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, func

from backend.database import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ============================================================================
# Response Models
# ============================================================================

class DailyPerformance(BaseModel):
    """Daily performance summary"""
    date: str
    total_pnl: float
    total_pnl_pct: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    winrate: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_win: float
    avg_loss: float
    equity_end: float


class StrategyAttribution(BaseModel):
    """Performance attribution by strategy"""
    strategy_id: str
    strategy_name: str
    total_pnl: float
    total_pnl_pct: float
    trades_count: int
    winrate: float
    profit_factor: float
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    contribution_to_portfolio: float


class ModelComparison(BaseModel):
    """Model performance comparison"""
    model_type: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: float
    directional_accuracy: float
    total_predictions: int
    status: str
    trained_at: str


class RiskMetrics(BaseModel):
    """Current risk metrics"""
    current_drawdown_pct: float
    max_drawdown_pct: float
    peak_equity: float
    current_equity: float
    global_winrate: float
    consecutive_losses: int
    days_since_profit: int
    risk_mode: str
    position_count: int
    total_exposure: float
    leverage_avg: float


class OpportunityTrend(BaseModel):
    """Opportunity score trends"""
    timestamp: str
    symbol: str
    composite_score: float
    trend_score: float
    volatility_score: float
    liquidity_score: float
    volume_24h: float


# ============================================================================
# Database Dependency
# ============================================================================

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/daily", response_model=List[DailyPerformance])
async def get_daily_performance(
    days: int = Query(30, ge=1, le=365, description="Number of days to fetch"),
    db: Session = Depends(get_db)
):
    """
    Get daily performance summary for the last N days.
    
    Returns PnL, trade counts, winrate, and profit factor by day.
    """
    try:
        query = text("""
            SELECT 
                DATE(timestamp) as date,
                SUM(realized_pnl) as total_pnl,
                SUM(realized_pnl_pct) as total_pnl_pct,
                COUNT(*) as trades_count,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                MAX(realized_pnl) as largest_win,
                MIN(realized_pnl) as largest_loss,
                AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
                MAX(equity_after) as equity_end
            FROM trade_logs
            WHERE timestamp >= datetime('now', :period)
            AND status = 'closed'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT :days
        """)
        
        result = db.execute(query, {"period": f"-{days} days", "days": days}).fetchall()
        
        daily_performance = []
        for row in result:
            date, total_pnl, total_pnl_pct, trades, wins, losses, max_win, max_loss, avg_win, avg_loss, equity = row
            
            winrate = (wins / trades) if trades > 0 else 0.0
            
            # Calculate profit factor
            total_wins = abs(avg_win * wins) if avg_win and wins > 0 else 0.001
            total_losses = abs(avg_loss * losses) if avg_loss and losses > 0 else 0.001
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
            
            daily_performance.append(DailyPerformance(
                date=str(date),
                total_pnl=float(total_pnl or 0),
                total_pnl_pct=float(total_pnl_pct or 0),
                trades_count=int(trades),
                winning_trades=int(wins),
                losing_trades=int(losses),
                winrate=float(winrate),
                profit_factor=float(profit_factor),
                largest_win=float(max_win or 0),
                largest_loss=float(max_loss or 0),
                avg_win=float(avg_win or 0),
                avg_loss=float(avg_loss or 0),
                equity_end=float(equity or 0)
            ))
        
        return daily_performance
        
    except Exception as e:
        logger.error(f"[Analytics] Error fetching daily performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=List[StrategyAttribution])
async def get_strategy_attribution(
    days: int = Query(30, ge=1, le=365, description="Lookback period in days"),
    db: Session = Depends(get_db)
):
    """
    Get performance attribution by strategy.
    
    Shows which strategies are contributing to overall performance.
    """
    try:
        query = text("""
            SELECT 
                COALESCE(strategy_id, 'unknown') as strategy_id,
                COALESCE(strategy_id, 'Unknown Strategy') as strategy_name,
                SUM(realized_pnl) as total_pnl,
                SUM(realized_pnl_pct) as total_pnl_pct,
                COUNT(*) as trades_count,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                0.0 as max_drawdown_pct
            FROM trade_logs
            WHERE timestamp >= datetime('now', :period)
            AND status = 'closed'
            GROUP BY strategy_id
            ORDER BY total_pnl DESC
        """)
        
        result = db.execute(query, {"period": f"-{days} days"}).fetchall()
        
        # Calculate total portfolio PnL for contribution calculation
        total_portfolio_pnl = sum(row[2] for row in result if row[2])
        
        attributions = []
        for row in result:
            strategy_id, strategy_name, total_pnl, total_pnl_pct, trades, wins, max_dd = row
            
            winrate = (wins / trades) if trades > 0 else 0.0
            contribution = (total_pnl / total_portfolio_pnl * 100) if total_portfolio_pnl else 0.0
            
            # Profit factor calculation (simplified)
            profit_factor = (wins / (trades - wins)) if (trades - wins) > 0 else 0.0
            
            attributions.append(StrategyAttribution(
                strategy_id=str(strategy_id),
                strategy_name=str(strategy_name),
                total_pnl=float(total_pnl or 0),
                total_pnl_pct=float(total_pnl_pct or 0),
                trades_count=int(trades),
                winrate=float(winrate),
                profit_factor=float(profit_factor),
                sharpe_ratio=None,  # TODO: Calculate from returns
                max_drawdown_pct=float(max_dd or 0),
                contribution_to_portfolio=float(contribution)
            ))
        
        return attributions
        
    except Exception as e:
        logger.error(f"[Analytics] Error fetching strategy attribution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelComparison])
async def get_model_comparison(
    db: Session = Depends(get_db)
):
    """
    Compare performance of all AI models (XGBoost, LightGBM, N-HiTS, PatchTST).
    
    Shows accuracy, precision, RMSE, and other metrics for each model version.
    """
    try:
        query = text("""
            SELECT 
                model_type,
                version,
                status,
                metrics,
                trained_at
            FROM model_artifacts
            WHERE status IN ('active', 'shadow')
            ORDER BY model_type, trained_at DESC
        """)
        
        result = db.execute(query).fetchall()
        
        comparisons = []
        for row in result:
            model_type, version, status, metrics_json, trained_at = row
            
            # Parse metrics JSON
            import json
            metrics = json.loads(metrics_json) if metrics_json else {}
            
            comparisons.append(ModelComparison(
                model_type=str(model_type),
                version=str(version),
                accuracy=float(metrics.get('accuracy', 0)),
                precision=float(metrics.get('precision', 0)),
                recall=float(metrics.get('recall', 0)),
                f1_score=float(metrics.get('f1_score', 0)),
                rmse=float(metrics.get('rmse', 0)),
                directional_accuracy=float(metrics.get('directional_accuracy', 0)),
                total_predictions=int(metrics.get('total_predictions', 0)),
                status=str(status),
                trained_at=str(trained_at)
            ))
        
        return comparisons
        
    except Exception as e:
        logger.error(f"[Analytics] Error fetching model comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk", response_model=RiskMetrics)
async def get_risk_metrics(
    db: Session = Depends(get_db)
):
    """
    Get current risk metrics and system health indicators.
    
    Shows drawdown, winrate, consecutive losses, risk mode, and exposure.
    """
    try:
        # Get drawdown and equity
        equity_query = text("""
            SELECT 
                MAX(equity_after) as peak_equity,
                MIN(equity_after) as trough_equity,
                (SELECT equity_after FROM trade_logs ORDER BY timestamp DESC LIMIT 1) as current_equity
            FROM trade_logs
            WHERE timestamp >= datetime('now', '-30 days')
        """)
        
        equity_result = db.execute(equity_query).fetchone()
        peak, trough, current = equity_result if equity_result else (None, None, None)
        
        # Handle None values
        peak = peak or 0
        trough = trough or 0
        current = current or 0
        
        current_dd = ((peak - current) / peak * 100) if peak > 0 else 0.0
        max_dd = ((peak - trough) / peak * 100) if peak > 0 else 0.0
        
        # Get winrate
        winrate_query = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM trade_logs
            WHERE timestamp >= datetime('now', '-30 days')
            AND status = 'closed'
        """)
        
        winrate_result = db.execute(winrate_query).fetchone()
        total_trades, wins = winrate_result if winrate_result else (0, 0)
        
        # Handle None values
        total_trades = total_trades or 0
        wins = wins or 0
        
        global_winrate = (wins / total_trades) if total_trades > 0 else 0.0
        
        # Get consecutive losses
        consecutive_query = text("""
            SELECT realized_pnl
            FROM trade_logs
            WHERE status = 'closed'
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        recent_trades = db.execute(consecutive_query).fetchall()
        consecutive_losses = 0
        for (pnl,) in recent_trades:
            if pnl is not None and pnl <= 0:
                consecutive_losses += 1
            else:
                break
        
        # Get days since profit
        last_profit_query = text("""
            SELECT MAX(DATE(timestamp)) as last_profit_date
            FROM trade_logs
            WHERE realized_pnl > 0
            AND status = 'closed'
        """)
        
        last_profit_result = db.execute(last_profit_query).fetchone()
        last_profit_date = last_profit_result[0] if last_profit_result else None
        
        if last_profit_date:
            from datetime import date
            days_since_profit = (date.today() - datetime.fromisoformat(last_profit_date).date()).days
        else:
            days_since_profit = 999
        
        # Get current positions
        position_query = text("""
            SELECT COUNT(*), SUM(qty * COALESCE(entry_price, price)), 1.0
            FROM trade_logs
            WHERE status = 'open'
        """)
        
        position_result = db.execute(position_query).fetchone()
        position_count, total_exposure, avg_leverage = position_result if position_result else (0, 0, 1)
        
        # Get risk mode from PolicyStore
        risk_mode = "NORMAL"  # Default
        try:
            from backend.services.policy_store import get_policy_store
            policy_store = get_policy_store()
            if policy_store:
                risk_mode = policy_store.get("risk_mode", "NORMAL")
        except Exception:
            pass
        
        return RiskMetrics(
            current_drawdown_pct=float(current_dd),
            max_drawdown_pct=float(max_dd),
            peak_equity=float(peak or 0),
            current_equity=float(current or 0),
            global_winrate=float(global_winrate),
            consecutive_losses=int(consecutive_losses),
            days_since_profit=int(days_since_profit),
            risk_mode=str(risk_mode),
            position_count=int(position_count or 0),
            total_exposure=float(total_exposure or 0),
            leverage_avg=float(avg_leverage or 0)
        )
        
    except Exception as e:
        logger.error(f"[Analytics] Error fetching risk metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/opportunities", response_model=List[OpportunityTrend])
async def get_opportunity_trends(
    request: Request,
    hours: int = Query(24, ge=1, le=168, description="Lookback period in hours"),
    symbol: Optional[str] = Query(None, description="Filter by symbol (optional)")
):
    """
    Get opportunity score trends over time.
    
    Shows how opportunity scores have changed for symbols.
    Requires Redis OpportunityStore to be enabled.
    """
    try:
        # Get OpportunityRanker from app state
        opportunity_ranker = getattr(request.app.state, 'opportunity_ranker', None)
        
        if not opportunity_ranker:
            raise HTTPException(
                status_code=503,
                detail="OpportunityRanker not available. Enable with QT_OPPORTUNITY_RANKER_ENABLED=true"
            )
        
        # Compute fresh rankings (this is fast, just scores)
        rankings_dict = opportunity_ranker.update_rankings()
        
        if not rankings_dict:
            return []
        
        # Filter by symbol if provided
        if symbol:
            rankings_dict = {k: v for k, v in rankings_dict.items() if k == symbol}
        
        # Convert to OpportunityTrend format (simplified - only current data)
        trends = []
        for sym, score in rankings_dict.items():
            trends.append(OpportunityTrend(
                timestamp=datetime.utcnow().isoformat(),
                symbol=sym,
                composite_score=score,
                trend_score=score * 0.4,  # Placeholder breakdown
                volatility_score=score * 0.3,
                liquidity_score=score * 0.3,
                volume_24h=0.0  # Not available in current structure
            ))
        
        return trends
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Analytics] Error fetching opportunity trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def analytics_health():
    """Health check endpoint for analytics API"""
    return {
        "status": "healthy",
        "service": "analytics_api",
        "timestamp": datetime.utcnow().isoformat()
    }

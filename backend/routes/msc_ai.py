"""
Meta Strategy Controller (MSC AI) API Endpoints

REST API for monitoring and controlling the MSC AI system.

Endpoints:
- GET /api/msc/status - Current MSC AI status and policy
- GET /api/msc/history - Historical policy changes
- POST /api/msc/evaluate - Trigger manual evaluation
- GET /api/msc/health - System health metrics

Author: Quantum Trader Team
Date: 2025-11-30
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.database import SessionLocal
from backend.services.msc_ai_integration import (
    get_msc_controller,
    run_msc_evaluation,
    QuantumPolicyStoreMSC
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/msc",
    tags=["MSC AI"],
    responses={404: {"description": "Not found"}}
)


def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/status")
async def get_msc_status(db: Session = Depends(get_db)) -> Dict:
    """
    Get current MSC AI status and active policy.
    
    Returns:
        Current policy, system health, and controller status
    """
    try:
        policy_store = QuantumPolicyStoreMSC()
        current_policy = policy_store.read_policy()
        
        if not current_policy:
            return {
                "status": "inactive",
                "message": "MSC AI has not run yet",
                "policy": None,
                "system_health": None
            }
        
        # Get system health from controller
        controller = get_msc_controller()
        health = controller.metrics_repo.get_system_health()
        
        # Count LIVE strategies
        strategies = controller.strategy_repo.list_live_strategies()
        
        return {
            "status": "active",
            "policy": {
                "risk_mode": current_policy["risk_mode"],
                "max_risk_per_trade_pct": current_policy["max_risk_per_trade"] * 100,
                "max_positions": current_policy["max_positions"],
                "min_confidence_pct": current_policy["global_min_confidence"] * 100,
                "max_daily_trades": current_policy.get("max_daily_trades", 30),
                "active_strategies_count": len(current_policy.get("allowed_strategies", [])),
                "allowed_strategies": current_policy.get("allowed_strategies", []),
                "updated_at": current_policy.get("updated_at")
            },
            "system_health": {
                "drawdown_pct": health.current_drawdown,
                "winrate_pct": health.global_winrate * 100,
                "equity_slope_pct_per_day": health.equity_slope_pct_per_day,
                "regime": health.regime.value,
                "volatility": health.volatility,
                "consecutive_losses": health.consecutive_losses,
                "days_since_profit": health.days_since_profit
            },
            "available_strategies": len(strategies),
            "controller_config": {
                "evaluation_period_days": controller.evaluation_period_days,
                "min_strategies": controller.min_strategies,
                "max_strategies": controller.max_strategies
            }
        }
        
    except Exception as e:
        logger.error(f"[MSC API] Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_msc_history(
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get historical MSC AI policy changes.
    
    Args:
        limit: Maximum number of records to return (max 200)
        
    Returns:
        List of historical policy records
    """
    try:
        query = text("""
            SELECT 
                id,
                risk_mode,
                max_risk_per_trade,
                max_positions,
                min_confidence,
                max_daily_trades,
                allowed_strategies,
                system_drawdown,
                system_winrate,
                created_at
            FROM msc_policies
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        
        results = db.execute(query, {"limit": limit}).fetchall()
        
        history = []
        for row in results:
            import json
            history.append({
                "id": row[0],
                "risk_mode": row[1],
                "max_risk_per_trade_pct": row[2] * 100,
                "max_positions": row[3],
                "min_confidence_pct": row[4] * 100,
                "max_daily_trades": row[5],
                "active_strategies_count": len(json.loads(row[6])) if row[6] else 0,
                "allowed_strategies": json.loads(row[6]) if row[6] else [],
                "system_drawdown_pct": row[7],
                "system_winrate_pct": row[8] * 100 if row[8] else 0.0,
                "created_at": row[9]
            })
        
        # Detect mode changes
        mode_changes = []
        for i in range(len(history) - 1):
            if history[i]["risk_mode"] != history[i + 1]["risk_mode"]:
                mode_changes.append({
                    "from_mode": history[i + 1]["risk_mode"],
                    "to_mode": history[i]["risk_mode"],
                    "timestamp": history[i]["created_at"],
                    "trigger": f"DD: {history[i]['system_drawdown_pct']:.1f}%, WR: {history[i]['system_winrate_pct']:.1f}%"
                })
        
        return {
            "total_records": len(history),
            "history": history,
            "mode_changes": mode_changes,
            "current_mode": history[0]["risk_mode"] if history else None
        }
        
    except Exception as e:
        logger.error(f"[MSC API] Failed to get history: {e}", exc_info=True)
        # Return empty history if table doesn't exist yet
        return {
            "total_records": 0,
            "history": [],
            "mode_changes": [],
            "current_mode": None,
            "message": "No history available yet"
        }


@router.post("/evaluate")
async def trigger_msc_evaluation() -> Dict:
    """
    Manually trigger MSC AI evaluation.
    
    Returns:
        Evaluation results and updated policy
    """
    try:
        logger.info("[MSC API] Manual evaluation triggered via API")
        result = run_msc_evaluation()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "message": "MSC AI evaluation completed",
            "evaluation": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MSC API] Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_system_health() -> Dict:
    """
    Get detailed system health metrics used by MSC AI.
    
    Returns:
        Comprehensive health metrics
    """
    try:
        controller = get_msc_controller()
        health = controller.metrics_repo.get_system_health()
        
        return {
            "status": "healthy",
            "metrics": {
                "drawdown": {
                    "current_pct": health.current_drawdown,
                    "threshold_defensive": 5.0,
                    "threshold_normal": 3.0,
                    "status": "critical" if health.current_drawdown > 5 else "warning" if health.current_drawdown > 3 else "good"
                },
                "winrate": {
                    "current_pct": health.global_winrate * 100,
                    "threshold_aggressive": 60.0,
                    "threshold_normal": 50.0,
                    "threshold_defensive": 45.0,
                    "status": "excellent" if health.global_winrate > 0.6 else "good" if health.global_winrate > 0.5 else "warning"
                },
                "equity_trend": {
                    "slope_pct_per_day": health.equity_slope_pct_per_day,
                    "status": "bullish" if health.equity_slope_pct_per_day > 0.3 else "bearish" if health.equity_slope_pct_per_day < -0.3 else "flat"
                },
                "market_regime": {
                    "current": health.regime.value,
                    "volatility": health.volatility,
                    "favorable": health.regime.value in ["BULL_TRENDING", "BEAR_TRENDING"]
                },
                "risk_signals": {
                    "consecutive_losses": health.consecutive_losses,
                    "days_since_profit": health.days_since_profit,
                    "warning_threshold": 5,
                    "status": "critical" if health.consecutive_losses >= 5 else "warning" if health.consecutive_losses >= 3 else "ok"
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[MSC API] Failed to get health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_strategy_rankings() -> Dict:
    """
    Get current strategy rankings from MSC AI scoring algorithm.
    
    Returns:
        Ranked list of all LIVE strategies with scores
    """
    try:
        controller = get_msc_controller()
        
        # Get all LIVE strategies
        strategies = controller.strategy_repo.list_live_strategies()
        
        if not strategies:
            return {
                "total_strategies": 0,
                "rankings": [],
                "message": "No LIVE strategies found"
            }
        
        # Score each strategy
        scorer = controller.scorer
        current_regime = controller.metrics_repo.get_system_health().regime
        
        rankings = []
        for strategy in strategies:
            stats = controller.strategy_repo.get_strategy_stats(
                strategy.strategy_id,
                period_days=controller.evaluation_period_days
            )
            
            if stats:
                score = scorer.score_strategy(strategy, stats, current_regime)
                
                rankings.append({
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.strategy_name,
                    "score": round(score.total_score, 3),
                    "metrics": {
                        "profit_factor": round(stats.profit_factor, 2),
                        "winrate_pct": round(stats.winrate * 100, 1),
                        "avg_R_multiple": round(stats.avg_R_multiple, 2),
                        "total_trades": stats.total_trades,
                        "max_drawdown_pct": round(stats.max_drawdown * 100, 2),
                        "total_pnl": round(stats.total_pnl, 2)
                    },
                    "score_breakdown": {
                        "profit_factor_score": round(score.pf_score, 3),
                        "winrate_score": round(score.wr_score, 3),
                        "drawdown_score": round(score.dd_score, 3),
                        "volume_score": round(score.volume_score, 3),
                        "regime_bonus": round(score.regime_bonus, 3)
                    },
                    "regime_compatibility": strategy.regime_compatibility,
                    "age_days": (datetime.now(timezone.utc) - strategy.created_at).days
                })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        # Mark which are currently selected
        policy_store = QuantumPolicyStoreMSC()
        current_policy = policy_store.read_policy()
        allowed = set(current_policy.get("allowed_strategies", [])) if current_policy else set()
        
        for rank in rankings:
            rank["currently_active"] = rank["strategy_id"] in allowed
        
        return {
            "total_strategies": len(rankings),
            "rankings": rankings,
            "current_regime": current_regime.value,
            "evaluation_period_days": controller.evaluation_period_days,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[MSC API] Failed to get strategy rankings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

"""
Strategy Router
Strategy management, backtests, optimization
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from .auth_router import verify_token

router = APIRouter(prefix="/strategy", tags=["Strategy Management"])

@router.get("/active")
def get_active_strategies():
    """Get all active trading strategies"""
    return {
        "strategies": [
            {
                "id": 1,
                "name": "momentum_v3",
                "status": "active",
                "allocation": 40.0,
                "daily_pnl": 1250.50,
                "win_rate": 0.72,
                "trades_today": 12
            },
            {
                "id": 2,
                "name": "mean_reversion",
                "status": "active",
                "allocation": 30.0,
                "daily_pnl": 850.25,
                "win_rate": 0.65,
                "trades_today": 8
            },
            {
                "id": 3,
                "name": "breakout_scalper",
                "status": "active",
                "allocation": 20.0,
                "daily_pnl": 350.00,
                "win_rate": 0.58,
                "trades_today": 18
            }
        ]
    }

@router.get("/{strategy_id}")
def get_strategy_details(strategy_id: int):
    """Get detailed information about a strategy"""
    return {
        "id": strategy_id,
        "name": "momentum_v3",
        "description": "Momentum-based strategy with ML enhancements",
        "version": "3.2.1",
        "status": "active",
        "allocation_percentage": 40.0,
        "capital_allocated": 400000,
        "performance": {
            "daily_pnl": 1250.50,
            "weekly_pnl": 6250.00,
            "monthly_pnl": 18750.00,
            "win_rate": 0.72,
            "sharpe_ratio": 2.1,
            "max_drawdown": 5.2
        },
        "parameters": {
            "lookback_period": 20,
            "entry_threshold": 0.65,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        },
        "last_updated": datetime.utcnow().isoformat()
    }

@router.get("/backtests")
def get_backtests():
    """Get backtest results"""
    return {
        "backtests": [
            {
                "id": 1,
                "strategy": "momentum_v3",
                "period": "2024-06-01 to 2024-12-01",
                "total_return": 42.5,
                "sharpe_ratio": 2.1,
                "max_drawdown": 8.3,
                "win_rate": 0.68,
                "status": "completed"
            }
        ]
    }

@router.get("/optimization")
def get_optimization_status():
    """Get strategy optimization status"""
    return {
        "enabled": True,
        "last_optimization": "2025-12-24T12:00:00Z",
        "next_scheduled": "2025-12-28T12:00:00Z",
        "strategies_in_optimization": []
    }

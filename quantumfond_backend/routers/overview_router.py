"""
Overview Router
Central command center - system status, key metrics, alerts
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime
from .auth_router import verify_token

router = APIRouter(prefix="/overview", tags=["Overview"])

@router.get("/dashboard")
def get_dashboard(payload: dict = Depends(verify_token)):
    """Get main dashboard data"""
    
    return {
        "system_health": {
            "status": "operational",
            "cpu": 45.2,
            "ram": 62.1,
            "uptime_hours": 72
        },
        "trading_status": {
            "mode": "LIVE",
            "active_positions": 12,
            "daily_pnl": 2450.75,
            "pnl_percentage": 2.45
        },
        "risk_summary": {
            "portfolio_var": 125000,
            "max_drawdown": 8.5,
            "sharpe_ratio": 1.85,
            "status": "healthy"
        },
        "ai_status": {
            "models_active": 4,
            "prediction_accuracy": 0.78,
            "last_update": datetime.utcnow().isoformat()
        },
        "alerts": [
            {"severity": "info", "message": "Market volatility increased", "timestamp": datetime.utcnow().isoformat()},
            {"severity": "warning", "message": "Position size near limit on BTC", "timestamp": datetime.utcnow().isoformat()}
        ]
    }

@router.get("/metrics")
def get_key_metrics(payload: dict = Depends(verify_token)):
    """Get key performance metrics"""
    
    return {
        "total_capital": 1000000,
        "deployed_capital": 450000,
        "available_capital": 550000,
        "daily_pnl": 2450.75,
        "weekly_pnl": 15230.50,
        "monthly_pnl": 52100.25,
        "win_rate": 0.68,
        "total_trades": 342,
        "active_strategies": 5
    }

@router.get("/status")
def get_system_status(payload: dict = Depends(verify_token)):
    """Get detailed system status"""
    
    return {
        "backend": "operational",
        "database": "operational",
        "redis": "operational",
        "ai_engine": "operational",
        "exchange_connection": "operational",
        "last_check": datetime.utcnow().isoformat()
    }

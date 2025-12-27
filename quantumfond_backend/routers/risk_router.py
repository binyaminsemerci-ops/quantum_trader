"""
Risk Router
Risk metrics, exposure analysis, circuit breakers
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import math
from .auth_router import verify_token

router = APIRouter(prefix="/risk", tags=["Risk Management"])

def safe_float(value, default=0.0):
    """Safely convert value to float with fallback"""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default

@router.get("/metrics")
def get_risk_metrics():
    """Get current risk metrics"""
    return {
        "portfolio_var": {
            "value": safe_float(125000),
            "confidence": safe_float(0.95),
            "horizon_days": 1
        },
        "exposure": {
            "gross_exposure": safe_float(450000),
            "net_exposure": safe_float(125000),
            "leverage": safe_float(0.45)
        },
        "drawdown": {
            "current": safe_float(3.2),
            "max": safe_float(8.5),
            "threshold": safe_float(15.0),
            "status": "healthy"
        },
        "concentration": {
            "top_position_percentage": 12.5,
            "top_3_positions_percentage": 32.1
        },
        "volatility": {
            "portfolio_vol": 18.5,
            "market_vol": 22.3
        },
        "last_update": datetime.utcnow().isoformat()
    }

@router.get("/limits")
def get_risk_limits():
    """Get risk limits and thresholds"""
    return {
        "max_position_size": 100000,
        "max_leverage": 3.0,
        "max_drawdown": 15.0,
        "max_daily_loss": 50000,
        "position_concentration_limit": 20.0,
        "var_limit": 150000
    }

@router.get("/circuit-breakers")
def get_circuit_breakers():
    """Get circuit breaker status"""
    return {
        "enabled": True,
        "breakers": [
            {
                "name": "daily_loss",
                "threshold": 50000,
                "current": 12450,
                "status": "armed",
                "triggered": False
            },
            {
                "name": "drawdown",
                "threshold": 15.0,
                "current": 8.5,
                "status": "armed",
                "triggered": False
            },
            {
                "name": "volatility_spike",
                "threshold": 50.0,
                "current": 22.3,
                "status": "armed",
                "triggered": False
            }
        ]
    }

@router.get("/exposure/breakdown")
def get_exposure_breakdown():
    """Get detailed exposure breakdown"""
    return {
        "by_symbol": [
            {"symbol": "BTCUSDT", "exposure": 125000, "percentage": 27.8},
            {"symbol": "ETHUSDT", "exposure": 87500, "percentage": 19.4},
            {"symbol": "SOLUSDT", "exposure": 62500, "percentage": 13.9}
        ],
        "by_strategy": [
            {"strategy": "momentum_v3", "exposure": 180000, "percentage": 40.0},
            {"strategy": "mean_reversion", "exposure": 135000, "percentage": 30.0}
        ],
        "total_gross": 450000,
        "total_net": 125000
    }

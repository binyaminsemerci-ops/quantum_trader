"""
Performance Router
Analytics, reporting, benchmarking
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import math
from .auth_router import verify_token

router = APIRouter(prefix="/performance", tags=["Performance Analytics"])

def safe_float(value, default=0.0):
    """Safely convert value to float with fallback"""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default

@router.get("/summary")
def get_performance_summary():
    """Get overall performance summary"""
    return {
        "total_return": {
            "daily": safe_float(2.45),
            "weekly": safe_float(8.12),
            "monthly": safe_float(15.23),
            "ytd": safe_float(52.10)
        },
        "sharpe_ratio": safe_float(1.85),
        "sortino_ratio": safe_float(2.12),
        "max_drawdown": safe_float(8.5),
        "win_rate": safe_float(0.68),
        "profit_factor": safe_float(1.92),
        "total_trades": 342,
        "winning_trades": 233,
        "losing_trades": 109,
        "average_win": safe_float(850.50),
        "average_loss": safe_float(-425.30),
        "largest_win": 5250.00,
        "largest_loss": -2150.00
    }

@router.get("/timeline")
def get_performance_timeline(
    period: str = "1M"
):
    """Get performance over time"""
    return {
        "period": period,
        "data_points": [
            {"date": "2025-12-01", "pnl": 2450.75, "cumulative": 45000},
            {"date": "2025-12-02", "pnl": 1850.50, "cumulative": 46850.50}
        ]
    }

@router.get("/benchmark")
def get_benchmark_comparison():
    """Compare performance against benchmarks"""
    return {
        "portfolio_return": 52.10,
        "benchmarks": [
            {"name": "BTC", "return": 45.5, "outperformance": 6.6},
            {"name": "ETH", "return": 38.2, "outperformance": 13.9},
            {"name": "S&P500", "return": 22.1, "outperformance": 30.0}
        ]
    }

@router.get("/attribution")
def get_return_attribution():
    """Get return attribution analysis"""
    return {
        "by_strategy": [
            {"strategy": "momentum_v3", "contribution": 21.5},
            {"strategy": "mean_reversion", "contribution": 18.2},
            {"strategy": "breakout_scalper", "contribution": 12.4}
        ],
        "by_symbol": [
            {"symbol": "BTCUSDT", "contribution": 28.3},
            {"symbol": "ETHUSDT", "contribution": 15.7},
            {"symbol": "SOLUSDT", "contribution": 8.1}
        ]
    }

@router.get("/reports/daily")
def get_daily_report():
    """Get daily performance report"""
    return {
        "date": datetime.utcnow().date().isoformat(),
        "pnl": 2450.75,
        "return_percentage": 2.45,
        "trades": 24,
        "win_rate": 0.71,
        "best_trade": {"symbol": "BTCUSDT", "pnl": 850.50},
        "worst_trade": {"symbol": "ETHUSDT", "pnl": -125.30},
        "generated_at": datetime.utcnow().isoformat()
    }

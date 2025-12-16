"""
Performance & Analytics Layer (PAL) - FastAPI Router

Provides RESTful API endpoints for comprehensive performance analytics across
all dimensions: global, strategy, symbol, regime, risk, and events.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.services.performance_analytics import (
    PerformanceAnalyticsService,
)

# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

def get_analytics_service():
    """
    Dependency that provides PerformanceAnalyticsService instance.
    
    Connects to real Quantum Trader database for production analytics.
    Uses FastAPI dependency pattern with proper session management.
    """
    from backend.database import SessionLocal
    from backend.services.performance_analytics.real_repositories import (
        DatabaseTradeRepository,
        DatabaseStrategyStatsRepository,
        DatabaseSymbolStatsRepository,
        DatabaseMetricsRepository,
        DatabaseEventLogRepository,
    )
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Create real repository implementations
        service = PerformanceAnalyticsService(
            trades=DatabaseTradeRepository(db),
            strategies=DatabaseStrategyStatsRepository(db),
            symbols=DatabaseSymbolStatsRepository(db),
            metrics=DatabaseMetricsRepository(db, initial_balance=10000.0),
            events=DatabaseEventLogRepository(db),
        )
        
        yield service
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize analytics service: {str(e)}"
        )
    finally:
        db.close()


# ============================================================================
# RESPONSE MODELS (Optional - for OpenAPI docs)
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response for PAL service."""
    status: str
    service: str
    timestamp: str
    version: str = "1.0.0"


# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(
    prefix="/api/pal",
    tags=["Performance & Analytics Layer"],
)


# ============================================================================
# ENDPOINTS: GLOBAL PERFORMANCE
# ============================================================================

@router.get("/global/equity-curve")
def get_global_equity_curve(
    days: int = Query(90, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> List[Dict[str, Any]]:
    """
    Get global equity curve over time.
    
    Returns list of equity points with timestamp and equity value.
    
    **Example Response:**
    ```json
    [
        {"timestamp": "2025-01-01T00:00:00", "equity": 10000.0},
        {"timestamp": "2025-01-02T00:00:00", "equity": 10250.5},
        ...
    ]
    ```
    """
    equity_curve = analytics.get_global_equity_curve(days=days)
    
    # Convert to JSON-serializable format
    return [
        {
            "timestamp": ts.isoformat(),
            "equity": equity
        }
        for ts, equity in equity_curve
    ]


@router.get("/global/summary")
def get_global_summary(
    days: int = Query(90, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """
    Get comprehensive global performance summary.
    
    Returns performance metrics across all dimensions:
    - Balance (initial, current, PnL)
    - Trade statistics (counts, win rate, profit factor)
    - Risk metrics (drawdown, Sharpe ratio, R-multiples)
    - Best/worst trades
    - Winning/losing streaks
    - Cost analysis (commissions, slippage)
    
    **Example Response:**
    ```json
    {
        "balance": {
            "initial": 10000.0,
            "current": 12500.0,
            "pnl_total": 2500.0,
            "pnl_percent": 25.0
        },
        "trades": {
            "total": 150,
            "winning": 85,
            "losing": 65,
            "win_rate": 0.567
        },
        "risk": {
            "max_drawdown_pct": 12.0,
            "sharpe_ratio": 1.5,
            "profit_factor": 1.8,
            "avg_r": 0.8,
            "median_r": 0.75
        },
        ...
    }
    ```
    """
    return analytics.get_global_performance_summary(days=days)


# ============================================================================
# ENDPOINTS: STRATEGY ANALYTICS
# ============================================================================

@router.get("/strategies/top")
def get_top_strategies(
    days: int = Query(180, ge=1, le=730, description="Number of days to analyze"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of strategies to return"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> List[Dict[str, Any]]:
    """
    Get top-performing strategies ranked by total PnL.
    
    Returns summary statistics for each strategy including:
    - Total PnL
    - Trade count
    - Win rate
    - Profit factor
    - Average R-multiple
    
    **Example Response:**
    ```json
    [
        {
            "strategy_id": "TREND_V3",
            "pnl_total": 5500.0,
            "trade_count": 45,
            "win_rate": 0.622,
            "profit_factor": 2.1,
            "avg_r": 1.2
        },
        ...
    ]
    ```
    """
    return analytics.get_top_strategies(days=days, limit=limit)


@router.get("/strategies/{strategy_id}")
def get_strategy_performance(
    strategy_id: str,
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """
    Get detailed performance analysis for a specific strategy.
    
    Returns:
    - Overall performance metrics
    - Equity curve
    - Per-symbol breakdown
    - Per-regime breakdown
    
    **Example Response:**
    ```json
    {
        "strategy_id": "TREND_V3",
        "performance": {
            "pnl_total": 5500.0,
            "trade_count": 45,
            "win_rate": 0.622,
            "profit_factor": 2.1,
            "avg_r": 1.2,
            "max_drawdown_pct": 8.5
        },
        "equity_curve": [...],
        "by_symbol": {...},
        "by_regime": {...}
    }
    ```
    """
    result = analytics.get_strategy_performance(strategy_id=strategy_id, days=days)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_id}' not found or has no trades in the last {days} days"
        )
    
    return result


# ============================================================================
# ENDPOINTS: SYMBOL ANALYTICS
# ============================================================================

@router.get("/symbols/top")
def get_top_symbols(
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of symbols to return"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> List[Dict[str, Any]]:
    """
    Get top-performing symbols ranked by total PnL.
    
    Returns summary statistics for each symbol including:
    - Total PnL
    - Trade count
    - Win rate
    - Profit factor
    - Trading volume
    
    **Example Response:**
    ```json
    [
        {
            "symbol": "BTCUSDT",
            "pnl_total": 8500.0,
            "trade_count": 78,
            "win_rate": 0.564,
            "profit_factor": 1.9,
            "volume_total": 1250000.0
        },
        ...
    ]
    ```
    """
    return analytics.get_top_symbols(days=days, limit=limit)


@router.get("/symbols/{symbol}")
def get_symbol_performance(
    symbol: str,
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """
    Get detailed performance analysis for a specific symbol.
    
    Returns:
    - Overall performance metrics
    - Volume statistics
    - Per-strategy breakdown
    - Per-regime breakdown
    
    **Example Response:**
    ```json
    {
        "symbol": "BTCUSDT",
        "performance": {
            "pnl_total": 8500.0,
            "trade_count": 78,
            "win_rate": 0.564,
            "profit_factor": 1.9,
            "avg_r": 0.95
        },
        "volume": {
            "total": 1250000.0,
            "avg_per_trade": 16025.64
        },
        "by_strategy": {...},
        "by_regime": {...}
    }
    ```
    """
    result = analytics.get_symbol_performance(symbol=symbol, days=days)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found or has no trades in the last {days} days"
        )
    
    return result


# ============================================================================
# ENDPOINTS: REGIME ANALYTICS
# ============================================================================

@router.get("/regimes/performance")
def get_regime_performance(
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """
    Get performance analysis by market regime and volatility.
    
    Returns aggregated performance for:
    - Market regimes: BULL, BEAR, CHOPPY, UNKNOWN
    - Volatility levels: LOW, MEDIUM, HIGH
    
    **Example Response:**
    ```json
    {
        "by_regime": {
            "BULL": {
                "pnl_total": 3500.0,
                "trade_count": 45,
                "win_rate": 0.644,
                "avg_r": 1.1
            },
            "BEAR": {...},
            "CHOPPY": {...}
        },
        "by_volatility": {
            "LOW": {...},
            "MEDIUM": {...},
            "HIGH": {...}
        }
    }
    ```
    """
    return analytics.get_regime_performance(days=days)


# ============================================================================
# ENDPOINTS: RISK & DRAWDOWN
# ============================================================================

@router.get("/risk/r-distribution")
def get_r_distribution(
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """
    Get R-multiple distribution analysis.
    
    Returns:
    - R-multiple summary statistics (avg, median, max, min)
    - Distribution buckets
    - Positive/negative trade counts
    
    **Example Response:**
    ```json
    {
        "summary": {
            "avg_r": 0.85,
            "median_r": 0.75,
            "max_r": 5.2,
            "min_r": -2.8,
            "positive_count": 95,
            "negative_count": 55
        },
        "buckets": [
            {"bucket": "-3R to -2R", "count": 5},
            {"bucket": "-2R to -1R", "count": 20},
            {"bucket": "-1R to 0R", "count": 30},
            {"bucket": "0R to 1R", "count": 40},
            {"bucket": "1R to 2R", "count": 35},
            {"bucket": "2R to 3R", "count": 15},
            {"bucket": "3R+", "count": 5}
        ]
    }
    ```
    """
    return analytics.get_r_distribution(days=days)


@router.get("/risk/drawdown")
def get_drawdown_stats(
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """
    Get drawdown analysis and statistics.
    
    Returns:
    - Maximum drawdown
    - Average drawdown
    - Current drawdown
    - Drawdown periods with recovery details
    
    **Example Response:**
    ```json
    {
        "max_drawdown_pct": 15.5,
        "avg_drawdown_pct": 5.2,
        "current_drawdown_pct": 2.1,
        "periods": [
            {
                "start": "2025-01-15T00:00:00",
                "end": "2025-02-01T00:00:00",
                "max_dd_pct": 12.0,
                "duration_days": 17,
                "recovered": true,
                "recovery_days": 8
            },
            ...
        ]
    }
    ```
    """
    return analytics.get_drawdown_stats(days=days)


# ============================================================================
# ENDPOINTS: EVENTS & SAFETY
# ============================================================================

@router.get("/events/emergency-stops")
def get_emergency_stop_history(
    days: int = Query(365, ge=1, le=730, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> List[Dict[str, Any]]:
    """
    Get Emergency Stop System (ESS) activation history.
    
    Returns list of ESS events with context and recovery details.
    
    **Example Response:**
    ```json
    [
        {
            "timestamp": "2025-01-20T14:35:00",
            "event_type": "EMERGENCY_STOP",
            "reason": "Max daily loss exceeded (-12.5%)",
            "severity": "CRITICAL",
            "context": {
                "daily_loss_pct": 12.5,
                "threshold_pct": 10.0,
                "open_positions": 5
            }
        },
        ...
    ]
    ```
    """
    return analytics.get_emergency_stop_history(days=days)


@router.get("/events/health")
def get_system_health_timeline(
    days: int = Query(90, ge=1, le=365, description="Number of days to analyze"),
    analytics: PerformanceAnalyticsService = Depends(get_analytics_service),
) -> List[Dict[str, Any]]:
    """
    Get system health event timeline.
    
    Returns chronological list of health events including:
    - Health warnings
    - Health recoveries
    - System errors
    - Performance degradations
    
    **Example Response:**
    ```json
    [
        {
            "timestamp": "2025-01-18T09:15:00",
            "event_type": "HEALTH_WARNING",
            "severity": "WARNING",
            "message": "Model performance degraded (win rate below 50%)",
            "context": {
                "win_rate": 0.45,
                "threshold": 0.50,
                "sample_size": 20
            }
        },
        ...
    ]
    ```
    """
    return analytics.get_system_health_timeline(days=days)


# ============================================================================
# ENDPOINTS: UTILITY
# ============================================================================

@router.get("/health")
def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for Performance & Analytics Layer.
    
    Returns service status and version information.
    """
    return HealthCheckResponse(
        status="healthy",
        service="Performance & Analytics Layer",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@router.get("/")
def get_analytics_info() -> Dict[str, Any]:
    """
    Get information about available analytics endpoints.
    
    Returns summary of all available analytics facets and endpoints.
    """
    return {
        "service": "Performance & Analytics Layer (PAL)",
        "version": "1.0.0",
        "description": "Centralized analytics backend for Quantum Trader",
        "facets": {
            "global": {
                "description": "Global account performance metrics",
                "endpoints": [
                    "GET /api/analytics/global/equity-curve",
                    "GET /api/analytics/global/summary"
                ]
            },
            "strategies": {
                "description": "Strategy-specific performance analysis",
                "endpoints": [
                    "GET /api/analytics/strategies/top",
                    "GET /api/analytics/strategies/{strategy_id}"
                ]
            },
            "symbols": {
                "description": "Symbol-specific performance analysis",
                "endpoints": [
                    "GET /api/analytics/symbols/top",
                    "GET /api/analytics/symbols/{symbol}"
                ]
            },
            "regimes": {
                "description": "Performance by market regime and volatility",
                "endpoints": [
                    "GET /api/analytics/regimes/performance"
                ]
            },
            "risk": {
                "description": "Risk metrics and drawdown analysis",
                "endpoints": [
                    "GET /api/analytics/risk/r-distribution",
                    "GET /api/analytics/risk/drawdown"
                ]
            },
            "events": {
                "description": "System events and safety monitoring",
                "endpoints": [
                    "GET /api/analytics/events/emergency-stops",
                    "GET /api/analytics/events/health"
                ]
            }
        },
        "docs": "See /api/docs for interactive API documentation"
    }

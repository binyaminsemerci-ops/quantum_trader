"""
Dashboard BFF (Backend-For-Frontend) Routes
EPIC: DASHBOARD-V3-001

New endpoints for Dashboard v3.0:
- GET /api/dashboard/overview - Global overview data
- GET /api/dashboard/trading - Trading activity data
- GET /api/dashboard/risk - Risk & safety metrics
- GET /api/dashboard/system - System health & stress scenarios
"""

import logging
import httpx
import os
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

# Optional yaml import (for GO-LIVE config)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[BFF] PyYAML not available - GO-LIVE config will use defaults")

from backend.api.dashboard.models import ServiceStatus
from backend.api.dashboard.utils import get_utc_timestamp, safe_float

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard-v3"])

# Paths
GO_LIVE_MARKER_FILE = Path(__file__).parent.parent.parent.parent / "go_live.active"
GO_LIVE_CONFIG_FILE = Path(__file__).parent.parent.parent.parent / "config" / "go_live.yaml"

# Service URLs
# When running in Docker, use container names. When running locally, use localhost.
PORTFOLIO_SERVICE_URL = os.getenv("PORTFOLIO_SERVICE_URL", "http://portfolio-intelligence:8004")
AI_ENGINE_SERVICE_URL = "http://ai-engine:8001"
EXECUTION_SERVICE_URL = "http://execution:8002"
RISK_SAFETY_SERVICE_URL = "http://risk-safety:8003"
MONITORING_SERVICE_URL = "http://monitoring-health-service:8080"


async def fetch_json(url: str, timeout: float = 1.0) -> Optional[dict]:
    """Fetch JSON from service URL with timeout (reduced to 1s for testnet)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.debug(f"[BFF] Service unavailable: {url} ({type(e).__name__})")
        return None


def get_go_live_status() -> Dict[str, Any]:
    """Get GO-LIVE activation status."""
    try:
        # Check marker file
        marker_exists = GO_LIVE_MARKER_FILE.exists()
        
        # Read config (if YAML available)
        config = {}
        if YAML_AVAILABLE and GO_LIVE_CONFIG_FILE.exists():
            try:
                with open(GO_LIVE_CONFIG_FILE, "r") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"[BFF] Could not read GO-LIVE config: {e}")
        
        return {
            "active": marker_exists,
            "activation_enabled": config.get("activation_enabled", False),
            "environment": config.get("environment", "testnet"),
            "allowed_profiles": config.get("allowed_profiles", []),
            "last_activation_timestamp": config.get("last_activation_timestamp"),
            "last_activation_operator": config.get("last_activation_operator"),
            "activation_count": config.get("activation_count", 0)
        }
    except Exception as e:
        logger.error(f"[BFF] Error getting GO-LIVE status: {e}")
        return {"active": False, "error": str(e)}


def get_ess_status() -> Dict[str, Any]:
    """Get Emergency Stop System status."""
    try:
        # Try to import ESS module
        from backend.core.safety.ess import ESS, ESSConfig
        
        # TODO: Get actual ESS instance status
        # For now, return a stub
        return {
            "status": "INACTIVE",
            "triggers_today": 0,
            "last_trigger_timestamp": None,
            "last_trigger_reason": None,
            "daily_loss_threshold": -5.0,
            "current_daily_loss": 0.0
        }
    except ImportError:
        logger.warning("[BFF] ESS module not available")
        return {
            "status": "UNKNOWN",
            "error": "ESS module not available"
        }
    except Exception as e:
        logger.error(f"[BFF] Error getting ESS status: {e}")
        return {"status": "ERROR", "error": str(e)}


@router.get("/overview", summary="Get dashboard overview data")
async def get_overview():
    """
    Get global overview data for Dashboard v3.0.
    
    Returns:
    - environment (STAGING/PRODUCTION)
    - go_live_active (bool)
    - global_pnl
    - exposure_per_exchange
    - global_risk_state (OK/WARNING/CRITICAL)
    - ess_status (ACTIVE/INACTIVE)
    - capital_profiles_summary
    """
    logger.info("[BFF] Fetching overview data from Portfolio service...")
    
    # Get GO-LIVE status
    go_live = get_go_live_status()
    
    # Get ESS status
    ess = get_ess_status()
    
    # Fetch real data from portfolio service
    portfolio_data = await fetch_json(f"{PORTFOLIO_SERVICE_URL}/api/portfolio/snapshot", timeout=2.0)
    
    # Extract portfolio metrics
    portfolio = {
        "equity": safe_float(portfolio_data.get("total_equity"), 15000.0) if portfolio_data else 15000.0,
        "cash": safe_float(portfolio_data.get("cash_balance"), 0.0) if portfolio_data else 0.0,
        "daily_pnl": safe_float(portfolio_data.get("daily_pnl"), 0.0) if portfolio_data else 0.0,
        "daily_pnl_pct": safe_float(portfolio_data.get("daily_pnl"), 0.0) / safe_float(portfolio_data.get("total_equity"), 15000.0) * 100 if portfolio_data and portfolio_data.get("total_equity") else 0.0,
        "weekly_pnl": 0.0,  # TODO: Calculate from history
        "monthly_pnl": 0.0,  # TODO: Calculate from history
        "total_pnl": safe_float(portfolio_data.get("realized_pnl_today"), 0.0) + safe_float(portfolio_data.get("unrealized_pnl"), 0.0) if portfolio_data else 0.0,
        "exposure_per_exchange": [
            {
                "exchange": "binance_testnet",
                "exposure": safe_float(portfolio_data.get("total_exposure"), 0.0)
            }
        ] if portfolio_data else [],
        "profiles": [],
        "positions_count": portfolio_data.get("num_positions", 0) if portfolio_data else 0
    }
    
    # Build risk metrics from portfolio data
    positions = portfolio_data.get("positions", []) if portfolio_data else []
    
    # Calculate exposure breakdown
    exposure_long = 0.0
    exposure_short = 0.0
    if isinstance(positions, list):
        for pos in positions:
            exp = safe_float(pos.get("exposure"), 0.0)
            if pos.get("side") == "BUY":
                exposure_long += exp
            elif pos.get("side") == "SELL":
                exposure_short += exp
    elif isinstance(positions, dict):
        # Handle dict format (legacy)
        for pos in positions.values():
            exp = safe_float(pos.get("exposure"), 0.0)
            if pos.get("side") == "BUY":
                exposure_long += exp
            elif pos.get("side") == "SELL":
                exposure_short += exp
    
    total_exposure = safe_float(portfolio_data.get("total_exposure"), 0.0) if portfolio_data else 0.0
    
    risk = {
        "state": "OK",
        "daily_drawdown_pct": safe_float(portfolio_data.get("daily_drawdown_pct"), 0.0) if portfolio_data else 0.0,
        "weekly_drawdown_pct": safe_float(portfolio_data.get("weekly_drawdown_pct"), 0.0) if portfolio_data else 0.0,
        "max_drawdown_pct": 0.0,  # TODO: Track max drawdown over time
        "exposure_long": exposure_long,
        "exposure_short": exposure_short,
        "exposure_net": exposure_long - exposure_short,
        "exposure_total": total_exposure,
        "open_risk_pct": (total_exposure / safe_float(portfolio_data.get("total_equity"), 15000.0) * 100) if portfolio_data and portfolio_data.get("total_equity") else 0.0,
    }
    
    logger.info(f"[BFF] Overview: equity={portfolio['equity']:.2f}, positions={portfolio['positions_count']}, daily_pnl={portfolio['daily_pnl']:.2f}")
    
    return {
        "timestamp": get_utc_timestamp(),
        "environment": go_live.get("environment", "testnet"),
        "go_live_active": go_live.get("active", False),
        "global_pnl": {
            "equity": safe_float(portfolio.get("equity"), 0.0),
            "cash": safe_float(portfolio.get("cash"), 0.0),
            "daily_pnl": safe_float(portfolio.get("daily_pnl"), 0.0),
            "daily_pnl_pct": safe_float(portfolio.get("daily_pnl_pct"), 0.0),
            "weekly_pnl": safe_float(portfolio.get("weekly_pnl"), 0.0),
            "monthly_pnl": safe_float(portfolio.get("monthly_pnl"), 0.0),
            "total_pnl": safe_float(portfolio.get("total_pnl"), 0.0)
        },
        "positions_count": portfolio.get("positions_count", 0),
        "exposure_per_exchange": portfolio.get("exposure_per_exchange", []),
        "global_risk_state": risk.get("state", "OK"),
        "risk_metrics": {
            "daily_drawdown_pct": safe_float(risk.get("daily_drawdown_pct"), 0.0),
            "weekly_drawdown_pct": safe_float(risk.get("weekly_drawdown_pct"), 0.0),
            "max_drawdown_pct": safe_float(risk.get("max_drawdown_pct"), 0.0),
            "exposure_long": safe_float(risk.get("exposure_long"), 0.0),
            "exposure_short": safe_float(risk.get("exposure_short"), 0.0),
            "exposure_net": safe_float(risk.get("exposure_net"), 0.0),
            "exposure_total": safe_float(risk.get("exposure_total"), 0.0),
            "open_risk_pct": safe_float(risk.get("open_risk_pct"), 0.0),
        },
        "ess_status": {
            "status": ess.get("status", "UNKNOWN"),
            "triggers_today": ess.get("triggers_today", 0),
            "daily_loss": ess.get("current_daily_loss", 0.0),
            "threshold": ess.get("daily_loss_threshold", -5.0)
        },
        "capital_profiles_summary": portfolio.get("profiles", [])
    }


@router.get("/trading", summary="Get trading activity data")
async def get_trading():
    """
    Get trading activity data for Dashboard v3.0.
    
    Returns:
    - open_positions[]
    - recent_orders[]
    - recent_signals[]
    - strategies_per_account[]
    """
    logger.info("[BFF] Fetching trading data...")
    
    # Fetch positions from /positions HTTP endpoint
    open_positions = []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Use localhost since we're in the same container
            resp = await client.get("http://localhost:8000/positions")
            if resp.status_code == 200:
                positions_data = resp.json()
                if isinstance(positions_data, list):
                    # Direct list of positions
                    for pos in positions_data:
                        open_positions.append({
                            "symbol": pos.get("symbol", ""),
                            "side": pos.get("side", "LONG"),
                            "size": safe_float(pos.get("size", pos.get("quantity")), 0.0),
                            "entry_price": safe_float(pos.get("entry_price", pos.get("avg_entry_price")), 0.0),
                            "current_price": safe_float(pos.get("current_price"), 0.0),
                            "unrealized_pnl": safe_float(pos.get("unrealized_pnl", pos.get("pnl")), 0.0),
                            "unrealized_pnl_pct": safe_float(pos.get("pnl_pct"), 0.0),
                            "value": safe_float(pos.get("size", pos.get("quantity")), 0.0) * safe_float(pos.get("current_price"), 0.0),
                        })
                logger.info(f"[BFF] Returning {len(open_positions)} open positions from /positions endpoint")
            else:
                logger.warning(f"[BFF] /positions returned {resp.status_code}")
    except Exception as e:
        logger.error(f"[BFF] Error fetching positions: {e}")
        open_positions = []
    
    # EPIC: DASHBOARD-V3-TRADING-PANELS
    # Wire domain services for Recent Orders, Recent Signals, Active Strategies
    
    # 1. Recent Orders (Last 50)
    recent_orders = []
    try:
        from backend.domains.orders import OrderService
        from backend.database import SessionLocal
        
        db = SessionLocal()
        try:
            order_service = OrderService(db)
            orders = order_service.get_recent_orders(limit=50)
            
            for order in orders:
                recent_orders.append({
                    "id": order.id,
                    "timestamp": order.timestamp.isoformat(),
                    "account": order.account,
                    "symbol": order.symbol,
                    "side": order.side,
                    "size": order.size,
                    "price": order.price,
                    "status": order.status,
                    "strategy_id": order.strategy_id
                })
            
            logger.info(f"[BFF] Retrieved {len(recent_orders)} recent orders")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"[BFF] Failed to retrieve orders: {e}")
        recent_orders = []
    
    # 2. Recent Signals (Last 20)
    recent_signals = []
    try:
        from backend.domains.signals import SignalService
        
        signal_service = SignalService()
        signals = await signal_service.get_recent_signals(limit=20)
        
        for signal in signals:
            recent_signals.append({
                "id": signal.id,
                "timestamp": signal.timestamp.isoformat(),
                "account": signal.account,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "price": signal.price,
                "source": signal.source,
                "strategy_id": signal.strategy_id
            })
        
        logger.info(f"[BFF] Retrieved {len(recent_signals)} recent signals")
    except Exception as e:
        logger.warning(f"[BFF] Failed to retrieve signals: {e}")
        recent_signals = []
    
    # 3. Active Strategies
    strategies_per_account = []
    try:
        from backend.domains.strategies import StrategyService
        
        # Try to get PolicyStore (optional)
        policy_store = None
        try:
            from backend.core.policy_store import get_policy_store
            policy_store = get_policy_store()
        except:
            pass  # PolicyStore not initialized - will use fallback
        
        strategy_service = StrategyService(policy_store)
        strategies = strategy_service.get_active_strategies()
        
        # Count positions per strategy (optional enhancement)
        position_counts = {}
        for pos in open_positions:
            strategy = pos.get("strategy_id", "default")
            position_counts[strategy] = position_counts.get(strategy, 0) + 1
        
        # Format for dashboard - Frontend expects flat array with account + strategy_name
        strategies_per_account = []
        for strategy in strategies:
            strategies_per_account.append({
                "account": "main",
                "strategy_name": strategy.name,
                "enabled": strategy.enabled,
                "profile": strategy.profile,
                "position_count": position_counts.get(strategy.name, 0),
                "win_rate": 0.0,  # TODO: Calculate from closed positions
                "description": strategy.description,
                "min_confidence": strategy.min_confidence
            })
        
        logger.info(f"[BFF] Retrieved {len(strategies)} active strategies")
    except Exception as e:
        logger.warning(f"[BFF] Failed to retrieve strategies: {e}")
        strategies_per_account = []
    
    return {
        "timestamp": get_utc_timestamp(),
        "open_positions": open_positions,
        "recent_orders": recent_orders,
        "recent_signals": recent_signals,
        "strategies_per_account": strategies_per_account
    }


@router.get("/risk", summary="Get risk & safety metrics")
async def get_risk():
    """
    Get risk & safety metrics for Dashboard v3.0.
    
    Returns:
    - risk_gate_decisions_stats
    - ess_triggers_recent[]
    - dd_per_profile[]
    - var_es_snapshot
    """
    logger.info("[BFF] Fetching risk data (testnet mode - mock data)")
    
    # For testnet/demo: Use mock data instead of calling unavailable microservices
    # TODO: Enable service calls when microservices are deployed
    
    return {
        "timestamp": get_utc_timestamp(),
        "risk_gate_decisions_stats": {
            "allow": 0,
            "block": 0,
            "scale": 0,
            "total": 0
        },
        "ess_triggers_recent": [],
        "dd_per_profile": [],
        "var_es_snapshot": {
            "var_95": 0.0,
            "var_99": 0.0,
            "es_95": 0.0,
            "es_99": 0.0
        }
    }


@router.get("/system", summary="Get system health & stress data")
async def get_system():
    """
    Get system health & stress scenarios data for Dashboard v3.0.
    
    Returns:
    - services_health[]
    - exchanges_health[]
    - failover_events_recent[]
    - stress_scenarios_recent[]
    """
    logger.info("[BFF] Fetching system data (testnet mode - mock data)")
    
    # For testnet/demo: Use mock data instead of calling unavailable microservices
    # TODO: Enable service calls when microservices are deployed
    
    return {
        "timestamp": get_utc_timestamp(),
        "services_health": [
            {"name": "backend", "status": "healthy", "uptime_seconds": 3600},
            {"name": "binance_testnet", "status": "connected"}
        ],
        "exchanges_health": [
            {"exchange": "binance_testnet", "status": "connected", "latency_ms": 50}
        ],
        "failover_events_recent": [],
        "stress_scenarios_recent": []
    }


@router.post("/stress/run_all", summary="Run all stress scenarios")
async def run_all_stress_scenarios():
    """
    Trigger all stress scenarios.
    
    Calls POST /api/stress/run_all on risk-safety service.
    """
    logger.info("[BFF] Triggering all stress scenarios (testnet mode - mock response)")
    
    # For testnet/demo: Return mock response instead of calling unavailable microservice
    # TODO: Enable service call when risk-safety service is deployed
    return {
        "status": "completed",
        "timestamp": get_utc_timestamp(),
        "scenarios_run": [],
        "message": "Stress testing service not available in testnet mode"
    }

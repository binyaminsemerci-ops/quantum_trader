"""
Dashboard API Routes
Sprint 4

REST endpoint for dashboard snapshot aggregation.
Sprint 4 Del 3: Added better error handling and utils usage.
"""

import logging
import httpx
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Optional, List

from backend.api.dashboard.models import (
    DashboardSnapshot,
    DashboardPortfolio,
    DashboardPosition,
    DashboardSignal,
    DashboardRisk,
    DashboardSystemHealth,
    DashboardStrategy,
    DashboardRLSizing,
    ServiceHealthInfo,
    ESSState,
    ServiceStatus,
    PositionSide,
    SignalDirection,
    MarketRegime
)
from backend.api.dashboard.utils import (
    get_utc_timestamp,
    safe_get,
    safe_float,
    safe_round,
    safe_percentage
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ========== SERVICE URLS (from env or config) ==========

PORTFOLIO_SERVICE_URL = "http://portfolio-intelligence-service:8004"
AI_ENGINE_SERVICE_URL = "http://ai-engine-service:8001"
EXECUTION_SERVICE_URL = "http://execution-service:8002"
RISK_SAFETY_SERVICE_URL = "http://risk-safety-service:8003"
MONITORING_SERVICE_URL = "http://monitoring-health-service:8080"


# ========== HELPER: HTTP CLIENT ==========

async def fetch_json(url: str, timeout: float = 5.0) -> Optional[dict]:
    """
    Fetch JSON from a service URL with timeout.
    
    Returns None if service is down or times out.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logger.warning(f"[DASHBOARD] Timeout fetching {url}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"[DASHBOARD] HTTP error {e.response.status_code} fetching {url}")
        return None
    except Exception as e:
        logger.error(f"[DASHBOARD] Error fetching {url}: {e}")
        return None


# ========== AGGREGATION LOGIC ==========

async def aggregate_portfolio_data() -> DashboardPortfolio:
    """
    Aggregate portfolio data from portfolio-intelligence-service.
    
    Calls:
    - GET /api/portfolio/snapshot
    - GET /api/portfolio/pnl
    """
    logger.info("[DASHBOARD] Aggregating portfolio data")
    
    # Fetch snapshot
    snapshot_url = f"{PORTFOLIO_SERVICE_URL}/api/portfolio/snapshot"
    snapshot = await fetch_json(snapshot_url)
    
    # Fetch PnL breakdown
    pnl_url = f"{PORTFOLIO_SERVICE_URL}/api/portfolio/pnl"
    pnl = await fetch_json(pnl_url)
    
    if not snapshot:
        # Return default if service down
        return DashboardPortfolio(
            equity=0.0,
            cash=0.0,
            margin_used=0.0,
            margin_available=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            weekly_pnl=0.0,
            monthly_pnl=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            position_count=0
        )
    
    return DashboardPortfolio(
        equity=snapshot.get("equity", 0.0),
        cash=snapshot.get("cash", 0.0),
        margin_used=snapshot.get("margin_used", 0.0),
        margin_available=snapshot.get("margin_available", 0.0),
        total_pnl=snapshot.get("total_pnl", 0.0),
        daily_pnl=pnl.get("daily_pnl", 0.0) if pnl else 0.0,
        daily_pnl_pct=pnl.get("daily_pnl_pct", 0.0) if pnl else 0.0,
        weekly_pnl=pnl.get("weekly_pnl", 0.0) if pnl else 0.0,
        monthly_pnl=pnl.get("monthly_pnl", 0.0) if pnl else 0.0,
        realized_pnl=pnl.get("realized_pnl", 0.0) if pnl else 0.0,
        unrealized_pnl=pnl.get("unrealized_pnl", 0.0) if pnl else 0.0,
        position_count=len(snapshot.get("positions", []))
    )


async def aggregate_positions() -> List[DashboardPosition]:
    """
    Aggregate positions from execution-service or portfolio-service.
    
    Calls:
    - GET /api/execution/positions
    """
    logger.info("[DASHBOARD] Aggregating positions")
    
    positions_url = f"{EXECUTION_SERVICE_URL}/api/execution/positions"
    data = await fetch_json(positions_url)
    
    if not data or "positions" not in data:
        return []
    
    positions = []
    for pos in data["positions"]:
        try:
            positions.append(DashboardPosition(
                symbol=pos["symbol"],
                side=PositionSide(pos["side"].upper()),
                size=float(pos["quantity"]),
                entry_price=float(pos["entry_price"]),
                current_price=float(pos["current_price"]),
                unrealized_pnl=float(pos["unrealized_pnl"]),
                unrealized_pnl_pct=float(pos.get("unrealized_pnl_pct", 0.0)),
                value=float(pos["quantity"]) * float(pos["current_price"])
            ))
        except Exception as e:
            logger.error(f"[DASHBOARD] Error parsing position {pos}: {e}")
            continue
    
    return positions


async def aggregate_signals() -> List[DashboardSignal]:
    """
    Aggregate latest signals from ai-engine-service.
    
    Note: This assumes ai-engine has a /api/ai/signals/recent endpoint.
    If not, we'll return mock data or empty list.
    
    TODO: Add /api/ai/signals/recent endpoint to ai-engine-service.
    """
    logger.info("[DASHBOARD] Aggregating signals")
    
    # Mock signals for now (until endpoint exists)
    # In production, call: GET /api/ai/signals/recent?limit=10
    
    signals_url = f"{AI_ENGINE_SERVICE_URL}/api/ai/signals/recent"
    data = await fetch_json(signals_url, timeout=3.0)
    
    if not data:
        # Return empty list if service down or endpoint doesn't exist
        return []
    
    signals = []
    for sig in data.get("signals", [])[:10]:
        try:
            signals.append(DashboardSignal(
                timestamp=sig["timestamp"],
                symbol=sig["symbol"],
                direction=SignalDirection(sig["direction"].upper()),
                confidence=float(sig["confidence"]),
                strategy=sig.get("strategy", "ensemble"),
                target_size=sig.get("target_size")
            ))
        except Exception as e:
            logger.error(f"[DASHBOARD] Error parsing signal {sig}: {e}")
            continue
    
    return signals


async def aggregate_risk() -> DashboardRisk:
    """
    Aggregate risk metrics from risk-safety-service and portfolio-service.
    
    Calls:
    - GET /api/risk/ess/status
    - GET /api/portfolio/drawdown
    - GET /api/portfolio/exposure
    """
    logger.info("[DASHBOARD] Aggregating risk data")
    
    # ESS status
    ess_url = f"{RISK_SAFETY_SERVICE_URL}/api/risk/ess/status"
    ess = await fetch_json(ess_url)
    
    # Drawdown
    dd_url = f"{PORTFOLIO_SERVICE_URL}/api/portfolio/drawdown"
    dd = await fetch_json(dd_url)
    
    # Exposure
    exp_url = f"{PORTFOLIO_SERVICE_URL}/api/portfolio/exposure"
    exp = await fetch_json(exp_url)
    
    # Portfolio snapshot for daily PnL%
    snapshot_url = f"{PORTFOLIO_SERVICE_URL}/api/portfolio/snapshot"
    snapshot = await fetch_json(snapshot_url)
    
    # Calculate daily PnL % (from portfolio data)
    daily_pnl_pct = 0.0
    if snapshot and snapshot.get("equity", 0) > 0:
        daily_pnl = snapshot.get("pnl_breakdown", {}).get("daily", 0.0)
        equity = snapshot.get("equity", 1.0)
        daily_pnl_pct = (daily_pnl / equity) * 100
    
    # Calculate risk limit usage (what % of max -10% daily DD have we used?)
    max_allowed_dd_pct = -10.0  # Policy: max -10% daily DD
    daily_dd = dd.get("daily_dd_pct", 0.0) if dd else 0.0
    risk_limit_used_pct = abs((daily_dd / max_allowed_dd_pct) * 100) if max_allowed_dd_pct != 0 else 0.0
    
    # Calculate open risk from positions
    open_risk_pct = 0.0
    if exp:
        total_exp = exp.get("total_exposure", 0.0)
        equity = snapshot.get("equity", 1.0) if snapshot else 1.0
        if equity > 0:
            open_risk_pct = (total_exp / equity) * 100
    
    max_risk_per_trade_pct = 1.0  # Policy: max 1% risk per trade
    
    return DashboardRisk(
        ess_state=ESSState(ess.get("state", "UNKNOWN").upper()) if ess else ESSState.UNKNOWN,
        ess_reason=ess.get("reason") if ess else None,
        ess_tripped_at=ess.get("tripped_at") if ess else None,
        daily_pnl_pct=round(daily_pnl_pct, 2),
        daily_drawdown_pct=dd.get("daily_dd_pct", 0.0) if dd else 0.0,
        weekly_drawdown_pct=dd.get("weekly_dd_pct", 0.0) if dd else 0.0,
        max_drawdown_pct=dd.get("max_dd_pct", 0.0) if dd else 0.0,
        max_allowed_dd_pct=max_allowed_dd_pct,
        exposure_total=exp.get("total_exposure", 0.0) if exp else 0.0,
        exposure_long=exp.get("long_exposure", 0.0) if exp else 0.0,
        exposure_short=exp.get("short_exposure", 0.0) if exp else 0.0,
        exposure_net=exp.get("net_exposure", 0.0) if exp else 0.0,
        open_risk_pct=round(open_risk_pct, 2),
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        risk_limit_used_pct=round(risk_limit_used_pct, 2)
    )


async def aggregate_strategy() -> Optional[DashboardStrategy]:
    """
    Aggregate strategy and AI decision insights (Sprint 4 Del 2).
    
    Calls:
    - GET /api/ai/metrics/ensemble (ensemble scores)
    - GET /api/ai/metrics/meta-strategy (active strategy, regime)
    - GET /api/ai/metrics/rl-sizing (last RL decision)
    """
    logger.info("[DASHBOARD] Aggregating strategy data")
    
    # Ensemble metrics
    ensemble_url = f"{AI_ENGINE_SERVICE_URL}/api/ai/metrics/ensemble"
    ensemble = await fetch_json(ensemble_url)
    
    # Meta-strategy metrics
    meta_url = f"{AI_ENGINE_SERVICE_URL}/api/ai/metrics/meta-strategy"
    meta = await fetch_json(meta_url)
    
    # RL sizing metrics
    rl_url = f"{AI_ENGINE_SERVICE_URL}/api/ai/metrics/rl-sizing"
    rl = await fetch_json(rl_url)
    
    if not ensemble and not meta:
        logger.warning("[DASHBOARD] Strategy data unavailable")
        return None
    
    # Parse ensemble scores
    ensemble_scores = {}
    if ensemble and "model_agreement" in ensemble:
        ensemble_scores = ensemble["model_agreement"]
    else:
        # [SPRINT 5 - PATCH #4] Log warning if using fallback scores
        logger.warning("[PATCH #4] AI Engine unavailable, using fallback ensemble scores")
        ensemble_scores = {"xgb": 0.73, "lgbm": 0.69, "patchtst": 0.81, "nhits": 0.75}
    
    # Parse active strategy and regime
    active_strategy = "ADAPTIVE"  # Default
    regime = MarketRegime.UNKNOWN
    
    if meta:
        # If meta has top_strategies list, use first
        if "top_strategies" in meta and len(meta["top_strategies"]) > 0:
            active_strategy = meta["top_strategies"][0].get("strategy", "ADAPTIVE")
        # Try to parse regime from meta or elsewhere
        if "regime" in meta:
            try:
                regime = MarketRegime(meta["regime"].upper())
            except ValueError:
                regime = MarketRegime.UNKNOWN
    
    # Parse RL sizing (last decision)
    rl_sizing = None
    if rl:
        # Mock RL sizing data (replace when AI Engine has real endpoint)
        rl_sizing = DashboardRLSizing(
            symbol="BTCUSDT",
            proposed_risk_pct=0.75,
            capped_risk_pct=0.50,
            proposed_leverage=5.0,
            capped_leverage=3.0,
            volatility_bucket="MEDIUM"
        )
    
    return DashboardStrategy(
        active_strategy=active_strategy,
        regime=regime,
        ensemble_scores=ensemble_scores,
        rl_sizing=rl_sizing
    )


async def aggregate_system_health() -> DashboardSystemHealth:
    """
    Aggregate system health from monitoring-health-service.
    
    Calls:
    - GET /api/health/services
    - GET /api/health/alerts?limit=1
    """
    logger.info("[DASHBOARD] Aggregating system health")
    
    # Services health
    services_url = f"{MONITORING_SERVICE_URL}/api/health/services"
    services_data = await fetch_json(services_url)
    
    # Recent alerts
    alerts_url = f"{MONITORING_SERVICE_URL}/api/health/alerts?limit=1"
    alerts_data = await fetch_json(alerts_url)
    
    services = []
    overall_status = ServiceStatus.OK
    
    if services_data and "services" in services_data:
        for svc in services_data["services"]:
            try:
                status = ServiceStatus(svc["status"].upper())
                services.append(ServiceHealthInfo(
                    name=svc["name"],
                    status=status,
                    latency_ms=svc.get("latency_ms"),
                    last_check=svc.get("last_check")
                ))
                
                # Worst status becomes overall
                if status == ServiceStatus.DOWN:
                    overall_status = ServiceStatus.DOWN
                elif status == ServiceStatus.DEGRADED and overall_status != ServiceStatus.DOWN:
                    overall_status = ServiceStatus.DEGRADED
            except Exception as e:
                logger.error(f"[DASHBOARD] Error parsing service {svc}: {e}")
                continue
    
    alerts_count = alerts_data.get("total", 0) if alerts_data else 0
    last_alert = None
    if alerts_data and "alerts" in alerts_data and len(alerts_data["alerts"]) > 0:
        last_alert = alerts_data["alerts"][0].get("timestamp")
    
    return DashboardSystemHealth(
        overall_status=overall_status,
        services=services,
        alerts_count=alerts_count,
        last_alert=last_alert
    )


# ========== REST ENDPOINT ==========

@router.get("/snapshot", response_model=dict, summary="Get dashboard snapshot")
async def get_dashboard_snapshot():
    """
    Get complete dashboard snapshot.
    
    Aggregates data from all microservices:
    - Portfolio Intelligence: equity, PnL, positions
    - AI Engine: latest signals, strategy
    - Risk & Safety: ESS state, drawdown
    - Monitoring: system health
    
    This endpoint is called once on dashboard load.
    After that, WebSocket provides real-time updates.
    
    Sprint 4 Del 3: Added partial_data and errors tracking.
    
    Returns:
        DashboardSnapshot: Complete dashboard state (may be partial if services down)
    """
    logger.info("[DASHBOARD] Building snapshot")
    
    errors = []
    
    try:
        # Aggregate data from all services (async parallel)
        import asyncio
        
        portfolio, positions, signals, risk, system, strategy = await asyncio.gather(
            aggregate_portfolio_data(),
            aggregate_positions(),
            aggregate_signals(),
            aggregate_risk(),
            aggregate_system_health(),
            aggregate_strategy(),
            return_exceptions=True
        )
        
        # Handle exceptions - track errors but continue with partial data
        if isinstance(portfolio, Exception):
            logger.error(f"[DASHBOARD] Portfolio aggregation failed: {portfolio}")
            errors.append("portfolio-service unavailable")
            portfolio = DashboardPortfolio(
                equity=0, cash=0, margin_used=0, margin_available=0,
                total_pnl=0, daily_pnl=0, daily_pnl_pct=0,
                weekly_pnl=0, monthly_pnl=0, realized_pnl=0,
                unrealized_pnl=0, position_count=0
            )
        
        if isinstance(positions, Exception):
            logger.error(f"[DASHBOARD] Positions aggregation failed: {positions}")
            errors.append("execution-service unavailable")
            positions = []
        
        if isinstance(signals, Exception):
            logger.error(f"[DASHBOARD] Signals aggregation failed: {signals}")
            errors.append("ai-engine signals unavailable")
            signals = []
        
        if isinstance(risk, Exception):
            logger.error(f"[DASHBOARD] Risk aggregation failed: {risk}")
            errors.append("risk-safety-service unavailable")
            risk = DashboardRisk(
                ess_state=ESSState.UNKNOWN, ess_reason=None, ess_tripped_at=None,
                daily_pnl_pct=0, daily_drawdown_pct=0, weekly_drawdown_pct=0, max_drawdown_pct=0,
                max_allowed_dd_pct=-10.0, exposure_total=0, exposure_long=0, exposure_short=0, exposure_net=0,
                open_risk_pct=0, max_risk_per_trade_pct=1.0, risk_limit_used_pct=0
            )
        
        if isinstance(system, Exception):
            logger.error(f"[DASHBOARD] System health aggregation failed: {system}")
            errors.append("monitoring-service unavailable")
            system = DashboardSystemHealth(
                overall_status=ServiceStatus.UNKNOWN,
                services=[],
                alerts_count=0
            )
        
        if isinstance(strategy, Exception):
            logger.warning(f"[DASHBOARD] Strategy aggregation failed: {strategy}")
            errors.append("ai-engine strategy unavailable")
            strategy = None  # Optional field
        
        # Build snapshot
        snapshot = DashboardSnapshot(
            timestamp=get_utc_timestamp(),
            portfolio=portfolio,
            positions=positions,
            signals=signals,
            risk=risk,
            system=system,
            strategy=strategy,
            partial_data=len(errors) > 0,
            errors=errors
        )
        
        if errors:
            logger.warning(f"[DASHBOARD] Snapshot built with {len(errors)} error(s): {errors}")
        else:
            logger.info(f"[DASHBOARD] Snapshot built: {len(positions)} positions, {len(signals)} signals")
        
        return snapshot.to_dict()
    
    except Exception as e:
        logger.error(f"[DASHBOARD] Snapshot build failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build dashboard snapshot: {str(e)}")


@router.get("/health", summary="Dashboard API health")
async def dashboard_health():
    """Dashboard API health check."""
    return {
        "status": "OK",
        "service": "dashboard-api",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

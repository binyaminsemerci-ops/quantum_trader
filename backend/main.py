from contextlib import asynccontextmanager
import signal
import time as _time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use explicit backend.* imports to avoid ModuleNotFoundError when launched with different CWDs
from backend.utils.logging import configure_logging, get_logger
from backend.utils.metrics import (
    router as metrics_router,
    add_metrics_middleware,
    update_model_info,
)
from backend.database import SessionLocal, ModelRegistry
from backend.routes import (
    trades,
    stats,
    chart,
    settings,
    binance,
    signals,
    prices,
    candles,
    stress,
    trade_logs,
    watchlist,
    health,
    layout,
    portfolio,
    trading,
    ws as dashboard_ws,
    enhanced_api,
    ai_trading,
)

try:
    from backend.alerts.evaluator import evaluator_loop, register_ws, unregister_ws  # type: ignore
except Exception:  # pragma: no cover - optional component

    async def evaluator_loop(*_args, **_kwargs):  # type: ignore
        return

    def register_ws(*_a, **_k):
        return

    def unregister_ws(*_a, **_k):
        return


import asyncio

configure_logging()
logger = get_logger(__name__)


async def _heartbeat_task():
    """Periodic heartbeat to monitor app liveness."""
    counter = 0
    while True:
        counter += 1
        logger.info("heartbeat #%d - app running normally", counter)
        # Optionally refresh active model info every N cycles (e.g. every 4 heartbeats = 60s)
        if counter % 4 == 0:
            try:
                with SessionLocal() as session:
                    row = (
                        session.query(ModelRegistry)
                        .filter(ModelRegistry.is_active == 1)
                        .first()
                    )
                    if row:
                        update_model_info(row.version, row.tag)
            except Exception as exc:  # pragma: no cover
                logger.debug("model info refresh failed: %s", exc)
        await asyncio.sleep(15)


def _setup_signal_handlers():
    """Install graceful shutdown handlers unless under pytest or non-main thread."""
    import os
    import threading

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if threading.current_thread() is not threading.main_thread():
        return

    def _handler(signum, _frame):  # pragma: no cover
        logger.warning("Received signal %s - graceful shutdown initiated", signum)
        return  # Let uvicorn orchestrate shutdown

    for sig_name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, sig_name):
            try:
                signal.signal(getattr(signal, sig_name), _handler)
            except Exception:  # pragma: no cover
                pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Setup signal handlers for debugging
    _setup_signal_handlers()
    logger.info("FastAPI lifespan startup - initializing background tasks")
    # Load active model info once at startup
    try:
        with SessionLocal() as session:
            row = (
                session.query(ModelRegistry)
                .filter(ModelRegistry.is_active == 1)
                .first()
            )
            if row:
                update_model_info(row.version, row.tag)
            else:
                update_model_info(None, None)
    except Exception as exc:  # pragma: no cover
        logger.debug("initial model info load failed: %s", exc)

    # Start background tasks
    tasks = []
    try:
        # Alert evaluator
        alerts_task = asyncio.create_task(evaluator_loop(5.0))
        tasks.append(alerts_task)
        app.state._alerts_task = alerts_task

        # Heartbeat monitor
        heartbeat_task = asyncio.create_task(_heartbeat_task())
        tasks.append(heartbeat_task)
        app.state._heartbeat_task = heartbeat_task

        logger.info("Background tasks started successfully")
    except Exception as exc:
        logger.exception("Failed to start background tasks: %s", exc)
        app.state._alerts_task = None
        app.state._heartbeat_task = None

    try:
        yield
    finally:
        logger.info("FastAPI lifespan shutdown - cleaning up background tasks")
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Background tasks cleanup completed")


app = FastAPI(lifespan=_lifespan)
add_metrics_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],  # Support both ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    logger.debug("health ping")
    return {"message": "Quantum Trader API is running"}


# Import moved to top of file

_START_TIME = _time.time()


@app.get("/api/v1/system/status")
@app.get("/system/status")
async def system_status():
    """System/runtime status for dashboard.

    Returns both legacy fields (uptime_seconds, evaluator_running) and a normalized
    shape the React UI expects (status, uptime_hours, cpu_usage, memory_usage, connections).
    """
    eval_running = False
    task = getattr(app.state, "_alerts_task", None)
    if task and not task.cancelled():
        eval_running = True

    # Optional config-derived flags
    has_binance_keys = None
    binance_testnet = None
    real_trading_enabled = None
    try:  # pragma: no cover - defensive
        from config.config import load_config  # type: ignore

        cfg = load_config()
        has_binance_keys = bool(cfg.binance_api_key and cfg.binance_api_secret)
        binance_testnet = bool(getattr(cfg, "binance_use_testnet", False))
        real_trading_enabled = bool(getattr(cfg, "enable_real_trading", False))
    except Exception:
        pass

    # Lightweight runtime metrics (best effort, but skip expensive ops for speed)
    cpu_usage = 0.0
    memory_usage = 0.0
    connections = 0
    try:
        import psutil  # type: ignore

        # Get actual system metrics
        cpu_usage = float(psutil.cpu_percent(interval=0.1))
        memory_usage = float(psutil.virtual_memory().percent)
        connections = len(psutil.net_connections())
    except Exception:  # psutil not installed or unsupported env
        pass

    uptime_seconds = round(_time.time() - _START_TIME, 2)

    # Compose response
    return {
        # Original / legacy fields
        "service": "quantum_trader_core",
        "version": getattr(app, "version", None),
        "uptime_seconds": uptime_seconds,
        "evaluator_running": eval_running,
        # Normalized / UI-friendly fields
        "status": "RUNNING" if eval_running else "IDLE",
        "uptime_hours": round(uptime_seconds / 3600, 2),
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "connections": connections,
        # Meta
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        # Config flags
        "has_binance_keys": has_binance_keys,
        "binance_testnet": binance_testnet,
        "real_trading_enabled": real_trading_enabled,
        # Active model (best-effort)
        "active_model": _get_active_model_meta(),
    }


def _get_active_model_meta():  # placed after definition for clarity
    try:
        with SessionLocal() as session:
            row = (
                session.query(ModelRegistry)
                .filter(ModelRegistry.is_active == 1)
                .first()
            )
            if row:
                # Attempt to parse metrics_json for sharpe / sortino / drawdown
                sharpe = None
                sortino = None
                max_dd = None
                try:
                    if row.metrics_json:
                        import json as _json

                        mj = _json.loads(row.metrics_json)
                        bt = (mj.get("backtest") or {}) if isinstance(mj, dict) else {}
                        sharpe = bt.get("sharpe")
                        sortino = bt.get("sortino")
                        max_dd = bt.get("max_drawdown")
                except Exception:  # pragma: no cover
                    pass
                return {
                    "version": row.version,
                    "tag": row.tag,
                    "id": row.id,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "max_drawdown": max_dd,
                }
    except Exception:
        return None
    return None


@app.get("/api/v1/model/active")
async def active_model():
    meta = _get_active_model_meta()
    if not meta:
        return {"active": False, "detail": "No active model"}
    return {"active": True, **meta}


# inkluder routere uten trailing slash-problemer
API_PREFIX = ""  # base (non-versioned)
V1_PREFIX = "/api/v1"  # compatibility / explicit version prefix


def _safe_include(router_obj, prefix: str):
    try:
        app.include_router(router_obj, prefix=prefix)
        logger.info(
            "Included router %s at prefix %s",
            getattr(router_obj, "tags", None) or router_obj,
            prefix,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed including router at %s: %s", prefix, exc)


for base in (API_PREFIX, V1_PREFIX):
    _safe_include(health.router, f"{base}")
    _safe_include(trades.router, f"{base}/trades")
    _safe_include(stats.router, f"{base}/stats")
    _safe_include(chart.router, f"{base}/chart")
    _safe_include(settings.router, f"{base}/settings")
    _safe_include(binance.router, f"{base}/binance")
    _safe_include(signals.router, f"{base}/signals")
    _safe_include(prices.router, f"{base}/prices")
    _safe_include(candles.router, f"{base}/candles")
    _safe_include(stress.router, f"{base}/stress")
    _safe_include(trade_logs.router, f"{base}")
    _safe_include(metrics_router, f"{base}/metrics")
    _safe_include(watchlist.router, f"{base}/watchlist")
    _safe_include(layout.router, f"{base}")
    _safe_include(portfolio.router, f"{base}/portfolio")
    _safe_include(trading.router, f"{base}/trading")
    _safe_include(ai_trading.router, f"{base}")
    _safe_include(enhanced_api.router, f"{base}")
    _safe_include(dashboard_ws.router, f"{base}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import (
    trades,
    stats,
    chart,
    settings,
    binance,
    signals,
    prices,
    candles,
    ws,
    trade_logs,
)
# from trading_bot.routes import router as trading_bot_router
from backend.exceptions import add_exception_handlers
from backend.logging_config import setup_logging
from backend.performance_monitor import add_monitoring_middleware
import os

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level=log_level)

app = FastAPI(
    title="Quantum Trader API",
    description="""
    ## AI-Powered Cryptocurrency Trading Platform

    Quantum Trader provides comprehensive cryptocurrency trading capabilities with AI-driven decision making.

    ### Features
    * **Real-time trading** - Execute trades on supported exchanges
    * **AI signals** - ML-powered buy/sell recommendations
    * **Performance monitoring** - Comprehensive metrics and analytics
    * **Risk management** - Position sizing and risk controls
    * **Historical analysis** - Backtest strategies and analyze performance

    ### Authentication
    Most endpoints require API authentication. Set your API keys via the `/api/settings` endpoint.

    ### Rate Limiting
    API requests are monitored for performance. See `/api/metrics/*` endpoints for current usage.

    ### Support
    - View logs via structured logging system
    - Monitor performance with built-in metrics
    - Database migrations handled via Alembic
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    terms_of_service="https://github.com/binyaminsemerci-ops/quantum_trader/blob/main/LICENSE",
    contact={
        "name": "Quantum Trader Team",
        "url": "https://github.com/binyaminsemerci-ops/quantum_trader",
        "email": "support@quantumtrader.dev",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/binyaminsemerci-ops/quantum_trader/blob/main/LICENSE",
    },
)

# Add exception handlers
add_exception_handlers(app)

# Add performance monitoring
add_monitoring_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # eller ["*"] for enkel test
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Quantum Trader API is running"}


@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    """Enhanced health endpoint with Phase 4 status."""
    response = {
        "status": "ok",
        "phases": {}
    }
    
    # Check Phase 4 APRL
    if hasattr(app.state, "aprl") and app.state.aprl:
        aprl_status = app.state.aprl.get_status()
        response["phases"]["phase4_aprl"] = {
            "active": True,
            "mode": aprl_status.get("mode", "UNKNOWN"),
            "metrics_tracked": aprl_status.get("performance_samples", 0),
            "policy_updates": aprl_status.get("policy_updates", 0)
        }
    else:
        response["phases"]["phase4_aprl"] = {
            "active": False,
            "reason": "APRL not initialized"
        }
    
    return response

@app.get("/health/phase4", tags=["System"], summary="Phase 4 APRL Detailed Status")
async def phase4_health():
    """Detailed Phase 4 Adaptive Policy Reinforcement status."""
    if not hasattr(app.state, "aprl") or not app.state.aprl:
        return {"error": "Phase 4 APRL not initialized"}
    
    return app.state.aprl.get_status()


# inkluder routere uten trailing slash-problemer
app.include_router(trades.router, prefix="/trades")
app.include_router(stats.router, prefix="/stats")
app.include_router(chart.router, prefix="/chart")
app.include_router(settings.router, prefix="/settings")
app.include_router(binance.router, prefix="/binance")
app.include_router(signals.router, prefix="/signals")
app.include_router(prices.router, prefix="/prices")
app.include_router(candles.router, prefix="/candles")
app.include_router(trade_logs.router)  # Trade logs (root level)
app.include_router(ws.router)  # WebSocket routes (no prefix needed)
# app.include_router(trading_bot_router, prefix="/trading-bot", tags=["Trading Bot"])


if __name__ == "__main__":
    # Local dev launcher: run `python backend/main.py` (from repo root) or `python main.py` inside backend dir.
    import uvicorn, os
    port = int(os.getenv("PORT", "8080"))  # Default moved to 8080 to avoid clashes with earlier processes
    # Disable reload by default to prevent log file churn causing infinite reload loops.
    reload_env = os.getenv("UVICORN_RELOAD", "0").lower()
    reload = reload_env in {"1", "true", "yes"}
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        # Note: if enabling reload and you still see rapid restarts, start with CLI flag:
        # uvicorn main:app --port 8080 --reload --reload-exclude logs --reload-exclude *.db
    )

# ============================================================================
# PHASE 4: ADAPTIVE POLICY REINFORCEMENT LAYER (APRL)
# ============================================================================

from services.adaptive_policy_reinforcement import AdaptivePolicyReinforcement
import logging
import os

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def initialize_phase4():
    """Initialize Phase 4 Adaptive Policy Reinforcement + Exit Brain v3"""
    try:
        logger.info("[PHASE 4] 🎯 Initializing Adaptive Policy Reinforcement...")
        
        # Get Phase 3 components if available (with safe fallbacks)
        safety_governor = getattr(app.state, "safety_governor", None)
        risk_brain = getattr(app.state, "risk_brain", None)
        event_bus = getattr(app.state, "event_bus", None)
        
        # Initialize APRL
        aprl = AdaptivePolicyReinforcement(
            governor=safety_governor,
            risk_brain=risk_brain,
            event_bus=event_bus,
            max_window=1000
        )
        
        app.state.aprl = aprl
        
        logger.info("[PHASE 4] ✅ Adaptive Policy Reinforcement initialized")
        logger.info(f"[APRL] Mode: {aprl.current_mode} | Window: {aprl.max_window} samples")
        
        if safety_governor:
            logger.info("[APRL] ✅ Safety Governor integration: ACTIVE")
        else:
            logger.info("[APRL] ⚠️ Safety Governor not available - limited functionality")
        
        if risk_brain:
            logger.info("[APRL] ✅ Risk Brain integration: ACTIVE")
        else:
            logger.info("[APRL] ⚠️ Risk Brain not available - limited functionality")
        
        if event_bus:
            logger.info("[APRL] ✅ EventBus integration: ACTIVE")
        else:
            logger.info("[APRL] ⚠️ EventBus not available - no event publishing")
        
        logger.info("[PHASE 4] 🎉 Real-time risk optimization ACTIVE")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize APRL: {e}", exc_info=True)
        app.state.aprl = None
    
    # ========================================================================
    # EXIT BRAIN V3 - DYNAMIC EXECUTOR (Adaptive ATR Stop-Loss)
    # ========================================================================
    try:
        from backend.config.exit_mode import is_exit_brain_live_fully_enabled
        
        if is_exit_brain_live_fully_enabled():
            logger.warning("[EXIT_BRAIN_V3] 🧠 Initializing ExitBrain v3 Dynamic Executor...")
            
            # Import executor components
            from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor
            from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
            from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
            from backend.services.execution.exit_order_gateway import submit_exit_order
            
            # Initialize Binance client for position source
            from binance.client import Client
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
            
            if not api_key or not api_secret:
                logger.error("[EXIT_BRAIN_V3] ❌ Missing BINANCE_API_KEY or BINANCE_API_SECRET")
                app.state.exit_brain_executor = None
                return
            
            binance_client = Client(api_key, api_secret, testnet=use_testnet)
            if use_testnet:
                binance_client.API_URL = 'https://testnet.binancefuture.com'
                logger.info("[EXIT_BRAIN_V3] Using Binance TESTNET")
            
            # Create gateway wrapper that injects client
            class ExitOrderGatewayWrapper:
                def __init__(self, client):
                    self.client = client
                
                async def submit_exit_order(self, **kwargs):
                    # Inject client if not provided
                    if 'client' not in kwargs:
                        kwargs['client'] = self.client
                    return await submit_exit_order(**kwargs)
            
            # Initialize components
            planner = ExitBrainV3()
            adapter = ExitBrainAdapter(planner=planner)
            exit_gateway = ExitOrderGatewayWrapper(binance_client)
            
            # Initialize executor
            loop_interval = float(os.getenv("EXIT_BRAIN_CHECK_INTERVAL_SEC", "10"))
            executor = ExitBrainDynamicExecutor(
                adapter=adapter,
                exit_order_gateway=exit_gateway,
                position_source=binance_client,
                loop_interval_sec=loop_interval,
                shadow_mode=False  # LIVE mode (controlled by config)
            )
            
            # Start monitoring loop
            await executor.start()
            
            app.state.exit_brain_executor = executor
            
            logger.warning(
                f"[EXIT_BRAIN_V3] ✅ Dynamic Executor STARTED "
                f"(interval={loop_interval}s, ATR-based adaptive stops ACTIVE)"
            )
            logger.warning("[EXIT_BRAIN_V3] 🎯 Monitoring all positions for:")
            logger.warning("[EXIT_BRAIN_V3]   • Adaptive ATR stop-loss (1.1x ATR)")
            logger.warning("[EXIT_BRAIN_V3]   • Trailing profit (0.3% offset)")
            logger.warning("[EXIT_BRAIN_V3]   • Profit harvesting (+0.4% trigger, 20% partial)")
            logger.warning("[EXIT_BRAIN_V3]   • Volatility governor (2% ATR threshold)")
            
        else:
            logger.info("[EXIT_BRAIN_V3] Not enabled (EXIT_MODE != EXIT_BRAIN_V3 or not LIVE_ROLLOUT=ENABLED)")
            app.state.exit_brain_executor = None
    
    except Exception as e:
        logger.error(f"[EXIT_BRAIN_V3] ❌ Failed to initialize: {e}", exc_info=True)
        app.state.exit_brain_executor = None


@app.on_event("shutdown")
async def shutdown_exit_brain():
    """Gracefully stop Exit Brain v3 executor on shutdown"""
    try:
        executor = getattr(app.state, "exit_brain_executor", None)
        if executor:
            logger.warning("[EXIT_BRAIN_V3] ⏹️  Stopping monitoring loop...")
            await executor.stop()
            logger.info("[EXIT_BRAIN_V3] ✅ Executor stopped gracefully")
    except Exception as e:
        logger.error(f"[EXIT_BRAIN_V3] Error during shutdown: {e}", exc_info=True)


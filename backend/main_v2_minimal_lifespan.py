import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import (
    trades,
    stats,
    chart,
    settings,
    binance,
    signals,
    prices,
    candles,
)
# from trading_bot.routes import router as trading_bot_router
from exceptions import add_exception_handlers
from logging_config import setup_logging
from performance_monitor import add_monitoring_middleware
import os

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level=log_level)


# STEP 1: MINIMAL LIFESPAN FUNCTION - Test that it executes
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Minimal lifespan function - tests async context execution"""
    try:
        # CRITICAL: Force async context switch
        await asyncio.sleep(0)
        print("=" * 60, flush=True)
        print("🔥 LIFESPAN FUNCTION STARTED (STEP 2 - AI IMPORT TEST) 🔥", flush=True)
        print("=" * 60, flush=True)
        print(f"[LIFESPAN] App instance: {app_instance}", flush=True)
        print("[LIFESPAN] ✅ Async context is active!", flush=True)
        
        # STEP 2: Runtime import of AISystemServices (prevents build-time execution)
        print("[LIFESPAN] 🔍 Attempting runtime import of AISystemServices...", flush=True)
        try:
            from backend.services.system_services import AISystemServices, get_ai_services
            print("[LIFESPAN] ✅ AISystemServices imported successfully!", flush=True)
            print(f"[LIFESPAN] AISystemServices class: {AISystemServices}", flush=True)
            
            # Initialize AI services
            print("[LIFESPAN] 🚀 Initializing AISystemServices...", flush=True)
            ai_services = AISystemServices()
            await ai_services.initialize()
            app_instance.state.ai_services = ai_services
            print("[LIFESPAN] ✅ AISystemServices initialized!", flush=True)
            
            # Get status
            status = ai_services.get_status()
            print(f"[LIFESPAN] 📊 Status: {status}", flush=True)
            
        except ImportError as e:
            print(f"[LIFESPAN] ⚠️ AISystemServices not available (ImportError): {e}", flush=True)
            app_instance.state.ai_services = None
        except Exception as e:
            print(f"[LIFESPAN] ❌ Error initializing AISystemServices: {e}", flush=True)
            app_instance.state.ai_services = None
        
        # Yield control back to FastAPI (server runs here)
        yield
        
        print("[LIFESPAN] 🛑 Shutting down...", flush=True)
    except Exception as e:
        print(f"[LIFESPAN] ❌ ERROR: {e}", flush=True)
        raise
    finally:
        print("[LIFESPAN] 🏁 Cleanup complete", flush=True)


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
    lifespan=lifespan,  # CRITICAL: Connect lifespan function to FastAPI
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


# inkluder routere uten trailing slash-problemer
app.include_router(trades.router, prefix="/trades")
app.include_router(stats.router, prefix="/stats")
app.include_router(chart.router, prefix="/chart")
app.include_router(settings.router, prefix="/settings")
app.include_router(binance.router, prefix="/binance")
app.include_router(signals.router, prefix="/signals")
app.include_router(prices.router, prefix="/prices")
app.include_router(candles.router, prefix="/candles")
# app.include_router(trading_bot_router, prefix="/trading-bot", tags=["Trading Bot"])

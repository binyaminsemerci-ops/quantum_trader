"""
Trading Bot Microservice - Main FastAPI application.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST

from microservices.trading_bot.simple_bot import SimpleTradingBot
from microservices.trading_bot.symbol_filter import fetch_top_symbols_by_volume, refresh_symbols_periodically

# [EPIC-OBS-001] Initialize observability (tracing, metrics, structured logging)
try:
    from backend.infra.observability import (
        init_observability,
        get_logger,
        instrument_fastapi,
        add_metrics_middleware,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    # Fallback to basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Initialize observability at module level (before service starts)
if OBSERVABILITY_AVAILABLE:
    init_observability(
        service_name="trading-bot",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

# Global bot instance
bot: SimpleTradingBot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - start/stop bot."""
    global bot
    
    logger.info("[TRADING-BOT-SERVICE] Starting up...")
    
    # Initialize Redis for direct event publishing
    event_bus = None
    try:
        import redis.asyncio as redis
        
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        redis_client = await redis.from_url(redis_url, decode_responses=False)  # decode_responses=False for xadd compatibility
        await redis_client.ping()  # Verify connection
        event_bus = {"redis": redis_client}  # Simple dict with redis client
        logger.info(f"[TRADING-BOT-SERVICE] ✅ Redis connected: {redis_url}")
    except Exception as e:
        logger.warning(f"[TRADING-BOT-SERVICE] ⚠️  Redis unavailable: {e}")
        event_bus = None
    
    # Create and start bot (polling mode with Redis for publishing)
    # Fetch top 50 symbols by 24h volume (mainnet/L1/L2 only)
    logger.info("[TRADING-BOT-SERVICE] 🔍 Fetching top 50 symbols by volume...")
    symbols_list = await fetch_top_symbols_by_volume(limit=50, min_volume_usd=10_000_000)
    logger.info(f"[TRADING-BOT-SERVICE] ✅ Monitoring {len(symbols_list)} symbols")
    
    bot = SimpleTradingBot(
        ai_engine_url=os.getenv("AI_ENGINE_URL", "http://ai-engine:8001"),
        symbols=symbols_list,
        check_interval_seconds=int(os.getenv("CHECK_INTERVAL_SECONDS", "60")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.50")),
        event_bus=event_bus,
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_secret=os.getenv("BINANCE_API_SECRET")
    )
    
    await bot.start()
    
    # Start background task for symbol refresh (every 6 hours)
    import asyncio
    refresh_task = asyncio.create_task(refresh_symbols_periodically(bot, refresh_interval_hours=6))
    
    logger.info("[TRADING-BOT-SERVICE] 🚀 Bot started")
    logger.info("[TRADING-BOT-SERVICE] 🔄 Symbol refresh scheduled every 6 hours")
    
    yield
    
    # Shutdown - Cancel refresh task
    logger.info("[TRADING-BOT-SERVICE] Shutting down...")
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    if bot:
        await bot.stop()
    if event_bus and "redis" in event_bus:
        await event_bus["redis"].close()
    logger.info("[TRADING-BOT-SERVICE] ✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Trading Bot Service",
    description="Autonomous trading signal generation",
    version="1.0.0",
    lifespan=lifespan
)

# [EPIC-OBS-001] Instrument FastAPI with tracing & metrics
if OBSERVABILITY_AVAILABLE:
    instrument_fastapi(app)
    add_metrics_middleware(app)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if bot is None:
        return JSONResponse(
            status_code=503,
            content={
                "service": "trading-bot",
                "status": "NOT_READY",
                "message": "Bot not initialized"
            }
        )
    
    status = bot.get_status()
    
    return {
        "service": "trading-bot",
        "status": "OK" if status["running"] else "STOPPED",
        "version": "1.0.0",
        "bot": status
    }


@app.post("/start")
async def start_bot():
    """Manually start bot."""
    if bot:
        await bot.start()
        return {"message": "Bot started"}
    return JSONResponse(status_code=503, content={"error": "Bot not initialized"})


@app.post("/stop")
async def stop_bot():
    """Manually stop bot."""
    if bot:
        await bot.stop()
        return {"message": "Bot stopped"}
    return JSONResponse(status_code=503, content={"error": "Bot not initialized"})


@app.get("/status")
async def get_status():
    """Get detailed bot status."""
    if bot:
        return bot.get_status()
    return JSONResponse(status_code=503, content={"error": "Bot not initialized"})


@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe."""
    if bot is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "ready": False}
        )
    
    status = bot.get_status()
    return {"status": "ready" if status["running"] else "not_ready", "ready": status["running"]}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

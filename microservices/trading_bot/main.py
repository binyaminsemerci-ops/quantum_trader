"""
Trading Bot Microservice - Main FastAPI application.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from microservices.trading_bot.simple_bot import SimpleTradingBot
from microservices.trading_bot.symbol_filter import fetch_top_symbols_by_volume, refresh_symbols_periodically

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    # 🔥 FIX: Use TRADING_SYMBOLS env var OR fetch top symbols by volume
    trading_symbols_env = os.getenv("TRADING_SYMBOLS")
    if trading_symbols_env:
        symbols_list = [s.strip() for s in trading_symbols_env.split(",") if s.strip()]
        logger.info(f"[TRADING-BOT-SERVICE] ✅ Using {len(symbols_list)} symbols from TRADING_SYMBOLS env: {symbols_list}")
    else:
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

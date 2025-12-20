#!/bin/bash
# Quick deploy trading bot - run this ON VPS

set -e

cd /root/quantum_trader

# Create directory
mkdir -p microservices/trading_bot

# Create simple_bot.py
cat > microservices/trading_bot/simple_bot.py << 'EOFBOT'
"""
Simple Trading Bot - Continuously generates trade signals from AI Engine.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, List
import aiohttp

logger = logging.getLogger(__name__)


class SimpleTradingBot:
    def __init__(
        self,
        ai_engine_url: str = "http://ai-engine:8001",
        symbols: List[str] = None,
        check_interval_seconds: int = 60,
        min_confidence: float = 0.70,
        event_bus = None
    ):
        self.ai_engine_url = ai_engine_url
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.check_interval = check_interval_seconds
        self.min_confidence = min_confidence
        self.event_bus = event_bus
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.signals_generated = 0
        
        logger.info(
            f"[TRADING-BOT] Initialized: {len(self.symbols)} symbols, "
            f"check every {check_interval_seconds}s, min_confidence={min_confidence:.0%}"
        )
    
    async def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._trading_loop())
        logger.info("[TRADING-BOT] ✅ Started")
    
    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[TRADING-BOT] ✅ Stopped")
    
    async def _trading_loop(self):
        try:
            while self.running:
                try:
                    for symbol in self.symbols:
                        await self._process_symbol(symbol)
                    await asyncio.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"[TRADING-BOT] Error in trading loop: {e}", exc_info=True)
                    await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("[TRADING-BOT] Trading loop cancelled")
    
    async def _process_symbol(self, symbol: str):
        try:
            market_data = await self._fetch_market_data(symbol)
            if not market_data:
                return
            
            prediction = await self._get_ai_prediction(symbol, market_data)
            if not prediction:
                return
            
            confidence = prediction.get("confidence", 0)
            if confidence < self.min_confidence:
                logger.debug(f"[TRADING-BOT] {symbol}: Low confidence {confidence:.2%}")
                return
            
            await self._publish_trade_signal(symbol, prediction, market_data)
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error processing {symbol}: {e}")
    
    async def _fetch_market_data(self, symbol: str) -> Optional[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        ticker = await resp.json()
                        return {
                            "symbol": symbol,
                            "price": float(ticker["lastPrice"]),
                            "volume_24h": float(ticker["volume"]),
                            "price_change_24h": float(ticker["priceChangePercent"]),
                            "high_24h": float(ticker["highPrice"]),
                            "low_24h": float(ticker["lowPrice"]),
                            "timestamp": datetime.utcnow().isoformat()
                        }
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error fetching market data: {e}")
            return None
    
    async def _get_ai_prediction(self, symbol: str, market_data: dict) -> Optional[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ai_engine_url}/api/ai/predict"
                payload = {"symbol": symbol, "market_data": market_data}
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.debug(f"[TRADING-BOT] {symbol}: {result.get('action')} @ {result.get('confidence', 0):.2%}")
                        return result
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error getting prediction: {e}")
            return None
    
    async def _publish_trade_signal(self, symbol: str, prediction: dict, market_data: dict):
        try:
            if not self.event_bus:
                return
            
            signal = {
                "symbol": symbol,
                "side": prediction.get("action", "HOLD").upper(),
                "confidence": prediction.get("confidence", 0),
                "entry_price": market_data["price"],
                "stop_loss": prediction.get("stop_loss", market_data["price"] * 0.98),
                "take_profit": prediction.get("take_profit", market_data["price"] * 1.02),
                "position_size_usd": prediction.get("position_size_usd", 100),
                "leverage": prediction.get("leverage", 5),
                "timestamp": datetime.utcnow().isoformat(),
                "model": prediction.get("model", "ensemble"),
                "reason": prediction.get("reason", "AI prediction")
            }
            
            if signal["side"] == "HOLD":
                return
            
            await self.event_bus.publish("trade.intent", signal)
            self.signals_generated += 1
            logger.info(
                f"[TRADING-BOT] 📡 {symbol} {signal['side']} @ ${signal['entry_price']:.2f} "
                f"(conf={signal['confidence']:.2%}, size=${signal['position_size_usd']:.0f})"
            )
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error publishing signal: {e}", exc_info=True)
    
    def get_status(self) -> dict:
        return {
            "running": self.running,
            "symbols": self.symbols,
            "check_interval_seconds": self.check_interval,
            "min_confidence": self.min_confidence,
            "signals_generated": self.signals_generated
        }
EOFBOT

# Create main.py
cat > microservices/trading_bot/main.py << 'EOFMAIN'
"""Trading Bot Microservice"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from microservices.trading_bot.simple_bot import SimpleTradingBot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bot: SimpleTradingBot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    logger.info("[TRADING-BOT-SERVICE] Starting...")
    
    try:
        from backend.utils.event_bus import EventBus
        event_bus = EventBus()
        await event_bus.connect()
        logger.info("[TRADING-BOT-SERVICE] ✅ EventBus connected")
    except Exception as e:
        logger.error(f"[TRADING-BOT-SERVICE] ❌ EventBus failed: {e}")
        event_bus = None
    
    bot = SimpleTradingBot(
        ai_engine_url=os.getenv("AI_ENGINE_URL", "http://ai-engine:8001"),
        symbols=os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT").split(","),
        check_interval_seconds=int(os.getenv("CHECK_INTERVAL_SECONDS", "60")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.70")),
        event_bus=event_bus
    )
    await bot.start()
    logger.info("[TRADING-BOT-SERVICE] 🚀 Started")
    yield
    logger.info("[TRADING-BOT-SERVICE] Shutting down...")
    if bot:
        await bot.stop()
    if event_bus:
        await event_bus.disconnect()

app = FastAPI(title="Trading Bot", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    if bot is None:
        return JSONResponse(status_code=503, content={"status": "NOT_READY"})
    status = bot.get_status()
    return {"service": "trading-bot", "status": "OK" if status["running"] else "STOPPED", "bot": status}

@app.get("/status")
async def get_status():
    return bot.get_status() if bot else JSONResponse(status_code=503, content={"error": "Not initialized"})
EOFMAIN

# Create Dockerfile
cat > microservices/trading_bot/Dockerfile << 'EOFDOCKER'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY microservices/trading_bot/requirements.txt /app/microservices/trading_bot/
RUN pip install --no-cache-dir -r /app/microservices/trading_bot/requirements.txt
COPY microservices/trading_bot/*.py /app/microservices/trading_bot/
COPY backend/utils/ /app/backend/utils/
COPY backend/__init__.py /app/backend/
RUN touch /app/microservices/__init__.py
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')"
CMD ["uvicorn", "microservices.trading_bot.main:app", "--host", "0.0.0.0", "--port", "8003"]
EOFDOCKER

# Create requirements.txt
cat > microservices/trading_bot/requirements.txt << 'EOFREQ'
fastapi==0.109.0
uvicorn[standard]==0.27.0
aiohttp==3.9.1
redis==5.0.1
pydantic==2.5.3
EOFREQ

echo "✅ Files created"

# Build
echo "🐳 Building..."
docker build -f microservices/trading_bot/Dockerfile -t quantum_trading_bot:latest .

# Stop old
docker stop quantum_trading_bot 2>/dev/null || true
docker rm quantum_trading_bot 2>/dev/null || true

# Start
echo "🚀 Starting..."
docker run -d \
  --name quantum_trading_bot \
  --network quantum_trader_quantum_trader \
  -e AI_ENGINE_URL=http://ai-engine:8001 \
  -e TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT \
  -e CHECK_INTERVAL_SECONDS=60 \
  -e MIN_CONFIDENCE=0.70 \
  -e REDIS_HOST=redis \
  -e REDIS_PORT=6379 \
  -p 8003:8003 \
  --restart unless-stopped \
  quantum_trading_bot:latest

sleep 5

echo ""
echo "📋 Logs:"
docker logs --tail 30 quantum_trading_bot

echo ""
echo "🏥 Health:"
curl -s http://localhost:8003/health | python3 -m json.tool

echo ""
echo "✅ DONE! Monitor: docker logs -f quantum_trading_bot"

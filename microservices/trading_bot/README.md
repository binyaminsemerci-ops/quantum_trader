# Trading Bot Microservice

**Autonomous trading signal generation from AI Engine predictions.**

## Architecture

```
Trading Bot → AI Engine (predictions) → EventBus (trade.intent) → Execution Service
```

## Features

- **Continuous Market Monitoring**: Checks configured symbols every 60s
- **AI-Driven Signals**: Calls AI Engine for predictions
- **Confidence Filtering**: Only publishes signals above min_confidence threshold
- **EventBus Integration**: Publishes trade.intent events
- **Zero Position Management**: Execution Service handles order placement
- **Paper Trading Ready**: Works with paper trading mode

## Configuration

Environment variables:

```bash
AI_ENGINE_URL=http://ai-engine:8001
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
CHECK_INTERVAL_SECONDS=60
MIN_CONFIDENCE=0.70
REDIS_HOST=redis
REDIS_PORT=6379
```

## API Endpoints

- `GET /health` - Health check
- `GET /status` - Bot status (running, signals generated)
- `POST /start` - Start bot
- `POST /stop` - Stop bot

## Trade Signal Format

Published to EventBus topic `trade.intent`:

```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "confidence": 0.85,
  "entry_price": 43250.50,
  "stop_loss": 42500.00,
  "take_profit": 44000.00,
  "position_size_usd": 100,
  "leverage": 5,
  "timestamp": "2024-12-13T10:30:00Z",
  "model": "ensemble",
  "reason": "Strong uptrend signal"
}
```

## Deployment

```bash
# Build
docker build -f microservices/trading_bot/Dockerfile -t quantum_trading_bot:latest .

# Run
docker run -d \
  --name quantum_trading_bot \
  --network quantum_trader_quantum_trader \
  -e AI_ENGINE_URL=http://ai-engine:8001 \
  -e TRADING_SYMBOLS=BTCUSDT,ETHUSDT \
  -e CHECK_INTERVAL_SECONDS=60 \
  -e MIN_CONFIDENCE=0.70 \
  -e REDIS_HOST=redis \
  -p 8003:8003 \
  quantum_trading_bot:latest
```

## Logs

```bash
docker logs -f quantum_trading_bot
```

Expected output:
```
[TRADING-BOT] Initialized: 3 symbols, check every 60s, min_confidence=70%
[TRADING-BOT] ✅ Started
[TRADING-BOT] 📡 Signal: BTCUSDT BUY @ $43250.50 (confidence=85%, size=$100)
```

## Integration with Execution Service

Execution Service listens for `trade.intent` events and:
1. Validates signal (risk checks)
2. Places order via Binance adapter
3. Creates Exit Brain V3 plan
4. Monitors position until exit

## Monitoring

Health check shows:
```json
{
  "service": "trading-bot",
  "status": "OK",
  "version": "1.0.0",
  "bot": {
    "running": true,
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "check_interval_seconds": 60,
    "min_confidence": 0.70,
    "signals_generated": 42
  }
}
```

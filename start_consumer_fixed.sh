#!/bin/bash
cd /home/qt/quantum_trader
source .env

docker run -d \
  --name quantum_trade_intent_consumer \
  --network quantum_trader_quantum_trader \
  -v /home/qt/quantum_trader/runner.py:/app/runner.py:ro \
  -v /home/qt/quantum_trader/backend/events/subscribers/trade_intent_subscriber.py:/app/backend/events/subscribers/trade_intent_subscriber.py:ro \
  -v /home/qt/quantum_trader/backend/services/execution/execution.py:/app/backend/services/execution/execution.py:ro \
  -e REDIS_HOST=quantum_redis \
  -e BINANCE_API_KEY=$BINANCE_API_KEY \
  -e BINANCE_API_SECRET=$BINANCE_API_SECRET \
  -e BINANCE_TESTNET=true \
  quantum_trader-backend:latest \
  python -u /app/runner.py

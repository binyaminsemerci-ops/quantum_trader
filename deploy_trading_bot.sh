#!/bin/bash

# Deploy Trading Bot to VPS
# Run this from WSL: bash /mnt/c/quantum_trader/deploy_trading_bot.sh

set -e

VPS="root@46.224.116.254"
BASE_DIR="/root/quantum_trader"

echo "=== Trading Bot Deployment ==="
echo ""

# 1. Create directory
echo "📁 Creating directory..."
ssh $VPS "mkdir -p $BASE_DIR/microservices/trading_bot"

# 2. Copy files
echo "📤 Copying files..."
scp /mnt/c/quantum_trader/microservices/trading_bot/*.py $VPS:$BASE_DIR/microservices/trading_bot/
scp /mnt/c/quantum_trader/microservices/trading_bot/*.txt $VPS:$BASE_DIR/microservices/trading_bot/
scp /mnt/c/quantum_trader/microservices/trading_bot/Dockerfile $VPS:$BASE_DIR/microservices/trading_bot/
scp /mnt/c/quantum_trader/microservices/trading_bot/README.md $VPS:$BASE_DIR/microservices/trading_bot/

# 3. Build Docker image
echo "🐳 Building Docker image..."
ssh $VPS "cd $BASE_DIR && docker build -f microservices/trading_bot/Dockerfile -t quantum_trading_bot:latest ."

# 4. Stop existing container (if any)
echo "🛑 Stopping existing container..."
ssh $VPS "docker stop quantum_trading_bot 2>/dev/null || true"
ssh $VPS "docker rm quantum_trading_bot 2>/dev/null || true"

# 5. Start new container
echo "🚀 Starting Trading Bot..."
ssh $VPS "docker run -d \
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
  quantum_trading_bot:latest"

# 6. Wait for startup
echo "⏳ Waiting for startup..."
sleep 5

# 7. Check logs
echo ""
echo "📋 Recent logs:"
ssh $VPS "docker logs --tail 20 quantum_trading_bot"

# 8. Health check
echo ""
echo "🏥 Health check:"
ssh $VPS "curl -s http://localhost:8003/health | python3 -m json.tool"

echo ""
echo "✅ Trading Bot deployed successfully!"
echo ""
echo "Monitor logs: ssh $VPS 'docker logs -f quantum_trading_bot'"

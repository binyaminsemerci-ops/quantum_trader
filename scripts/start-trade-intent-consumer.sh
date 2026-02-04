#!/bin/bash
# Start Trade.Intent Consumer Service
# Ensures trade.intent subscriber runs and auto-restarts

set -e

echo "🚀 Starting Trade.Intent Consumer Service"
echo "=========================================="

# Check if docker-compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Navigate to project root
cd /home/qt/quantum_trader || cd "$(dirname "$0")/.."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Using defaults."
fi

# Stop any existing instances
echo "📋 Stopping existing instances..."
docker compose -f docker-compose.trade-intent-consumer.yml down 2>/dev/null || true

# Pull/build latest images
echo "🔨 Building backend image..."
docker compose -f docker-compose.trade-intent-consumer.yml build backend 2>/dev/null || \
    docker compose build backend 2>/dev/null || \
    echo "⚠️  Using existing backend image"

# Start services
echo "🚀 Starting consumer service..."
docker compose -f docker-compose.trade-intent-consumer.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Verify services are running
echo ""
echo "📊 Service Status:"
echo "==================="
docker compose -f docker-compose.trade-intent-consumer.yml ps

# Check backend health
echo ""
echo "🏥 Health Check:"
echo "================"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend may still be starting..."
fi

# Check Redis
if docker exec quantum_redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Check for trade.intent subscriber in logs
echo ""
echo "🔍 Checking subscriber initialization..."
sleep 3
if docker logs quantum_backend 2>&1 | tail -100 | grep -q "TradeIntentSubscriber\|Phase 3.5"; then
    echo "✅ TradeIntentSubscriber initialized"
else
    echo "⚠️  Subscriber initialization not found in recent logs"
    echo "    Check: docker logs quantum_backend | grep -i trade_intent"
fi

# Show how to view logs
echo ""
echo "📝 View Logs:"
echo "============="
echo "  All logs:       docker logs -f quantum_backend"
echo "  Subscriber:     docker logs -f quantum_backend 2>&1 | grep -i trade_intent"
echo "  Redis stream:   docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent"

# Show consumer group status
echo ""
echo "📊 Consumer Group Status:"
echo "========================="
docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent 2>/dev/null || \
    echo "⚠️  Consumer group not yet created (will be created on first subscriber start)"

echo ""
echo "✅ Trade.Intent Consumer Service Started!"
echo ""
echo "🔄 Auto-restart is ENABLED (restart: always)"
echo "   The consumer will automatically start on system reboot."
echo ""
echo "📋 Management Commands:"
echo "  Stop:    docker compose -f docker-compose.trade-intent-consumer.yml down"
echo "  Restart: docker compose -f docker-compose.trade-intent-consumer.yml restart"
echo "  Status:  docker compose -f docker-compose.trade-intent-consumer.yml ps"

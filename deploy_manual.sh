#!/bin/bash
# Manual deployment script - bypass systemctl state corruption
# Usage: bash deploy_manual.sh

set -e

NETWORK="quantum_trader_quantum_trader"
REDIS_VOL="quantum_trader_redis_data"

echo "🚀 Starting deployment..."

# Ensure network exists
docker network inspect $NETWORK >/dev/null 2>&1 || docker network create $NETWORK

# Redis already running - check
if systemctl list-units | grep -q quantum_redis; then
    echo "✅ Redis already running"
else
    echo "Starting Redis..."
    docker run -d \
        --name quantum_redis \
        --network $NETWORK \
        -v $REDIS_VOL:/data \
        -p 6379:6379 \
        --health-cmd="redis-cli ping" \
        --health-interval=10s \
        redis:7-alpine redis-server --appendonly yes
fi

# Wait for Redis
echo "⏳ Waiting for Redis..."
until python3 quantum_redis redis-cli ping 2>/dev/null; do
    sleep 1
done
echo "✅ Redis healthy"

# Start services
SERVICES=(
    "ai-engine:quantum_ai_engine:8001"
    "ceo-brain:quantum_ceo_brain:8010"
    "strategy-brain:quantum_strategy_brain:8011"
    "risk-brain:quantum_risk_brain:8012"
    "binance-pnl-tracker:quantum_binance_pnl_tracker:8013"
    "cross-exchange:quantum_cross_exchange:8014"
    "exposure-balancer:quantum_exposure_balancer:8015"
    "meta-regime:quantum_meta_regime:8016"
    "model-federation:quantum_model_federation:8017"
    "portfolio-governance:quantum_portfolio_governance:8018"
    "portfolio-intelligence:quantum_portfolio_intelligence:8019"
    "quantumfond-frontend:quantum_quantumfond_frontend:3000"
    "retraining-worker:quantum_retraining_worker:8020"
    "rl-monitor:quantum_rl_monitor:8025"
    "rl-dashboard:quantum_rl_dashboard:8026"
    "strategic-memory:quantum_strategic_memory:8021"
    "strategic-evolution:quantum_strategic_evolution:8022"
    "trade-intent-consumer:quantum_trade_intent_consumer:8023"
)

for service in "${SERVICES[@]}"; do
    IFS=':' read -r image container port <<< "$service"
    
    # Remove if exists
    # No rm needed -f $container 2>/dev/null || true
    
    echo "Starting $container..."
    docker run -d \
        --name $container \
        --network $NETWORK \
        -e REDIS_HOST=quantum_redis \
        -e REDIS_URL=redis://quantum_redis:6379 \
        -e TESTNET=true \
        quantum_trader-$image
done

echo ""
echo "✅ Deployment complete!"
echo ""
systemctl list-units --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20


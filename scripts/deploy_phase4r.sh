#!/bin/bash
set -e

# ============================================================================
# Phase 4R+ — Meta-Regime Correlator Deployment Script
# ============================================================================
# Deploys Meta-Regime Correlator to VPS with validation and health checks
# Usage: ./deploy_phase4r.sh
# ============================================================================

echo "🚀  Starting Phase 4R+ — Meta-Regime Deployment"
echo "============================================================================"
cd /home/qt/quantum_trader

# 1️⃣  Update code
echo ""
echo "🔄  Pulling latest code..."
git pull origin main || echo "⚠️  Git pull failed (continuing anyway)"

# 2️⃣  Build Docker image
echo ""
echo "🏗️  Building meta-regime service..."
docker compose -f docker-compose.vps.yml build meta-regime

# 3️⃣  Start service
echo ""
echo "▶️  Starting meta-regime container..."
docker compose -f docker-compose.vps.yml up -d meta-regime
echo "⏳  Waiting 10 seconds for service initialization..."
sleep 10

# 4️⃣  Verify container is running
echo ""
echo "🔍  Checking container status..."
if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q meta_regime; then
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep meta_regime
    echo "✅  Meta-Regime container is running"
else
    echo "❌  Meta-Regime container not running"
    echo "📜  Checking logs for errors:"
    docker logs --tail 50 quantum_meta_regime
    exit 1
fi

# 5️⃣  Check Redis streams
echo ""
echo "📊  Checking Redis data structures..."
STREAM_LEN=$(docker exec redis redis-cli XLEN quantum:stream:meta.regime)
echo "   • Stream entries: $STREAM_LEN"

PREFERRED=$(docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime)
if [ -n "$PREFERRED" ]; then
    echo "   • Preferred regime: $PREFERRED"
else
    echo "   • Preferred regime: Not set yet (warming up)"
fi

# 6️⃣  Inject simulated regime data for testing
echo ""
echo "🧩  Injecting simulated regime data for testing..."
docker exec redis redis-cli XADD quantum:stream:meta.regime "*" \
    regime BULL pnl 0.42 volatility 0.015 trend 0.002 confidence 0.87 timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec redis redis-cli XADD quantum:stream:meta.regime "*" \
    regime BULL pnl 0.38 volatility 0.012 trend 0.003 confidence 0.91 timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec redis redis-cli XADD quantum:stream:meta.regime "*" \
    regime RANGE pnl 0.15 volatility 0.008 trend 0.000 confidence 0.82 timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec redis redis-cli XADD quantum:stream:meta.regime "*" \
    regime BEAR pnl -0.12 volatility 0.022 trend -0.004 confidence 0.79 timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec redis redis-cli XADD quantum:stream:meta.regime "*" \
    regime VOLATILE pnl -0.25 volatility 0.042 trend 0.001 confidence 0.73 timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "✅  Injected 5 sample regime observations"
echo "⏳  Waiting 5 seconds for correlator to process..."
sleep 5

# 7️⃣  Check Redis after injection
echo ""
echo "📊  Redis status after injection:"
STREAM_LEN_AFTER=$(docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime)
echo "   • Stream entries: $STREAM_LEN_AFTER"

PREFERRED_AFTER=$(docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime)
if [ -n "$PREFERRED_AFTER" ]; then
    echo "   • Preferred regime: $PREFERRED_AFTER"
else
    echo "   • Preferred regime: Still not set"
fi

# 8️⃣  AI Engine health check
echo ""
echo "🧠  Fetching AI Engine health status..."
if command -v jq &> /dev/null; then
    curl -s http://localhost:8001/health | jq '.metrics.meta_regime'
else
    echo "   (jq not installed, showing raw JSON)"
    curl -s http://localhost:8001/health | grep -A 10 '"meta_regime"'
fi

# 9️⃣  Check recent logs
echo ""
echo "📜  Recent meta-regime logs:"
echo "------------------------------------------------------------"
docker logs --tail 20 quantum_meta_regime
echo "------------------------------------------------------------"

# 🔟  Container health check
echo ""
echo "🏥  Container health status:"
docker inspect quantum_meta_regime --format='{{.State.Health.Status}}' 2>/dev/null || echo "No health check defined"

# 1️⃣1️⃣  Summary
echo ""
echo "============================================================================"
echo "🎯  PHASE 4R+ DEPLOYMENT COMPLETE"
echo "============================================================================"
echo ""
echo "✅  Service Status:"
echo "   • Container: quantum_meta_regime"
echo "   • Status: Running"
echo "   • Redis Stream: quantum:stream:meta.regime ($STREAM_LEN_AFTER entries)"
echo "   • Preferred Regime: ${PREFERRED_AFTER:-'Warming up...'}"
echo ""
echo "🔗  Integration Points:"
echo "   • AI Engine Health: http://localhost:8001/health"
echo "   • Portfolio Governance: quantum:governance:policy"
echo "   • RL Sizing Agent: Receives regime context"
echo "   • Exposure Balancer: Adjusts based on regime"
echo ""
echo "📊  Monitoring Commands:"
echo "   • Watch logs: docker logs -f quantum_meta_regime"
echo "   • Check regime: docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
echo "   • Stream length: docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
echo "   • Full health: curl -s http://localhost:8001/health | jq '.metrics.meta_regime'"
echo ""
echo "============================================================================"
echo "🚀  Meta-Regime Correlator is now actively analyzing market regimes!"
echo "============================================================================"

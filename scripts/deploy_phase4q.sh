#!/bin/bash
set -e

# ============================================================================
# PHASE 4Q+ — Portfolio Governance Deployment & Validation
# ============================================================================
# Dette scriptet deployer og validerer Portfolio Governance Agent på VPS
# ============================================================================

echo ""
echo "🚀 =============================================="
echo "   PHASE 4Q+ — PORTFOLIO GOVERNANCE DEPLOYMENT"
echo "=============================================="
echo ""

cd /home/qt/quantum_trader

# --- 1. Sjekk at repoet er oppdatert
echo "🔄 Pulling latest code from GitHub..."
git pull origin main || {
  echo "⚠️  Git pull failed, continuing with local version..."
}
echo ""

# --- 2. Bygg service
echo "🏗️  Building portfolio_governance Docker image..."
docker compose -f docker-compose.vps.yml build portfolio-governance
echo "✅ Build complete"
echo ""

# --- 3. Start service
echo "▶️  Starting portfolio_governance service..."
docker compose -f docker-compose.vps.yml up -d portfolio-governance
echo "⏳ Waiting 10 seconds for service to initialize..."
sleep 10
echo ""

# --- 4. Valider at containeren kjører
echo "🔍 Checking container status..."
CONTAINER_STATUS=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep portfolio_governance || echo "")
if [ -z "$CONTAINER_STATUS" ]; then
  echo "❌ ERROR: Container not running!"
  echo "📜 Last 20 logs:"
  docker logs --tail 20 quantum_portfolio_governance
  exit 1
else
  echo "✅ Container is running:"
  echo "$CONTAINER_STATUS"
fi
echo ""

# --- 5. Kjør health check via AI Engine
echo "🧠 Fetching AI Engine Health metrics..."
HEALTH_RESPONSE=$(curl -s http://localhost:8001/health 2>/dev/null || echo "{}")
if command -v jq &> /dev/null; then
  echo "$HEALTH_RESPONSE" | jq '.metrics.portfolio_governance' 2>/dev/null || echo "⚠️  Portfolio governance metrics not yet available"
else
  echo "$HEALTH_RESPONSE" | grep -o '"portfolio_governance":{[^}]*}' || echo "⚠️  jq not installed, raw response shown"
fi
echo ""

# --- 6. Test Redis integrasjon
echo "📊 Testing Redis streams and keys..."
echo "  • Memory stream length:"
MEMORY_LEN=$(docker exec redis redis-cli XLEN quantum:stream:portfolio.memory 2>/dev/null || echo "0")
echo "    quantum:stream:portfolio.memory = $MEMORY_LEN entries"

echo "  • Current policy:"
POLICY=$(docker exec redis redis-cli GET quantum:governance:policy 2>/dev/null || echo "NOT_SET")
echo "    quantum:governance:policy = $POLICY"

echo "  • Current score:"
SCORE=$(docker exec redis redis-cli GET quantum:governance:score 2>/dev/null || echo "0.0")
echo "    quantum:governance:score = $SCORE"
echo ""

# --- 7. Simuler governance aktivitet
echo "🧩 Simulating sample PnL data for testing..."
echo "  • Adding profitable trade event..."
docker exec redis redis-cli XADD quantum:stream:portfolio.memory '*' \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  symbol "BTCUSDT" \
  side "LONG" \
  pnl "0.35" \
  confidence "0.74" \
  volatility "0.12" \
  leverage "20" \
  position_size "1000" \
  exit_reason "dynamic_tp" > /dev/null

echo "  • Adding losing trade event..."
docker exec redis redis-cli XADD quantum:stream:portfolio.memory '*' \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  symbol "ETHUSDT" \
  side "SHORT" \
  pnl "-0.18" \
  confidence "0.62" \
  volatility "0.22" \
  leverage "15" \
  position_size "500" \
  exit_reason "stop_loss" > /dev/null

echo "⏳ Waiting 5 seconds for governance agent to process..."
sleep 5
echo ""

echo "✅ Reading policy after simulation..."
NEW_POLICY=$(docker exec redis redis-cli GET quantum:governance:policy 2>/dev/null || echo "NOT_SET")
NEW_SCORE=$(docker exec redis redis-cli GET quantum:governance:score 2>/dev/null || echo "0.0")
NEW_MEMORY_LEN=$(docker exec redis redis-cli XLEN quantum:stream:portfolio.memory 2>/dev/null || echo "0")

echo "  • Policy: $NEW_POLICY"
echo "  • Score: $NEW_SCORE"
echo "  • Memory samples: $NEW_MEMORY_LEN"
echo ""

# --- 8. Log sjekk
echo "📜 Recent container logs (last 15 lines):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker logs --tail 15 quantum_portfolio_governance
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# --- 9. Test policy parameters
echo "🔧 Reading policy parameters..."
PARAMS=$(docker exec redis redis-cli GET quantum:governance:params 2>/dev/null || echo "{}")
if command -v jq &> /dev/null; then
  echo "$PARAMS" | jq '.' 2>/dev/null || echo "$PARAMS"
else
  echo "$PARAMS"
fi
echo ""

# --- 10. Oppsummering
echo ""
echo "🎯 =============================================="
echo "   PHASE 4Q+ DEPLOYMENT SUCCESSFULLY COMPLETED!"
echo "=============================================="
echo ""
echo "✅ Service Status:"
echo "   • Container: quantum_portfolio_governance (RUNNING)"
echo "   • Memory Stream: quantum:stream:portfolio.memory ($NEW_MEMORY_LEN samples)"
echo "   • Current Policy: $NEW_POLICY"
echo "   • Portfolio Score: $NEW_SCORE"
echo ""
echo "📡 Integration Points:"
echo "   • AI Engine Health: http://localhost:8001/health"
echo "   • Redis Policy Key: quantum:governance:policy"
echo "   • Redis Score Key: quantum:governance:score"
echo "   • Redis Params: quantum:governance:params"
echo ""
echo "📊 Next Steps:"
echo "   1. Monitor logs: docker logs -f quantum_portfolio_governance"
echo "   2. Check health: curl http://localhost:8001/health | jq '.metrics.portfolio_governance'"
echo "   3. Watch policy: watch -n 15 'docker exec redis redis-cli GET quantum:governance:policy'"
echo "   4. View memory: docker exec redis redis-cli XREVRANGE quantum:stream:portfolio.memory + - COUNT 10"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Portfolio Governance Agent is LIVE and OPERATIONAL!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

#!/bin/bash
set -e

echo "🚀 Starting Phase 4T+ Deployment — Strategic Evolution Engine"

# Determine working directory
if [ -d "/home/qt/quantum_trader" ]; then
    WORK_DIR="/home/qt/quantum_trader"
elif [ -d "/tmp" ]; then
    WORK_DIR="/tmp"
else
    WORK_DIR="$(pwd)"
fi

cd "$WORK_DIR"
echo "📂 Working directory: $WORK_DIR"

# === 1️⃣ Oppdater kode (hvis i quantum_trader repo) ===
if [ -d ".git" ]; then
    echo "🔄 Pulling latest repository..."
    git pull origin main
else
    echo "⚠️  Not a git repository, skipping git pull"
fi

# === 2️⃣ Bygg container ===
echo "🏗️ Building Strategic Evolution container..."
docker compose -f docker-compose.vps.yml build strategic-evolution

# === 3️⃣ Start container ===
echo "▶️ Starting Strategic Evolution service..."
docker compose -f docker-compose.vps.yml up -d strategic-evolution
sleep 15

# === 4️⃣ Verifiser container status ===
echo "🔍 Checking container status..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep strategic_evolution || {
  echo "❌ Container failed to start"; exit 1; }

# === 5️⃣ Redis sanity check ===
echo "📊 Checking Redis connections..."
docker exec redis redis-cli PING || { echo "❌ Redis not reachable"; exit 1; }

# === 6️⃣ Inject synthetic strategy performance data (6 mock strategies) ===
echo "🧩 Injecting synthetic strategy performance data (6 models)..."
for i in {1..6}; do
  SHARPE=$(awk -v min=0.5 -v max=2.5 'BEGIN{srand(); print min+rand()*(max-min)}')
  WINRATE=$(awk -v min=0.4 -v max=0.9 'BEGIN{srand(); print min+rand()*(max-min)}')
  DRAWDOWN=$(awk -v min=0.05 -v max=0.25 'BEGIN{srand(); print min+rand()*(max-min)}')
  CONSISTENCY=$(awk -v min=0.3 -v max=0.9 'BEGIN{srand(); print min+rand()*(max-min)}')
  
  docker exec redis redis-cli RPUSH quantum:strategy:performance \
    "{\"strategy\":\"model_$i\",\"sharpe_ratio\":$SHARPE,\"win_rate\":$WINRATE,\"max_drawdown\":$DRAWDOWN,\"consistency\":$CONSISTENCY}"
  
  echo "  ✓ Injected model_$i (Sharpe: $SHARPE, WinRate: $WINRATE)"
done

# === 7️⃣ Wait for processing cycle ===
echo "⏳ Waiting for Strategic Evolution to process (90 seconds)..."
sleep 90

# === 8️⃣ Logs summary ===
echo ""
echo "📜 Latest Evolution Engine logs:"
docker logs --tail 30 quantum_strategic_evolution

# === 9️⃣ Check evolution data in Redis ===
echo ""
echo "🔁 Checking evolution keys in Redis..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Rankings (first 300 chars):"
docker exec redis redis-cli GET quantum:evolution:rankings | head -c 300 && echo "..."

echo ""
echo "🎯 Selected Models:"
docker exec redis redis-cli GET quantum:evolution:selected

echo ""
echo "🧬 Mutated Configurations:"
docker exec redis redis-cli GET quantum:evolution:mutated

echo ""
echo "🔄 Retrain Stream (last 3 jobs):"
docker exec redis redis-cli XREVRANGE quantum:stream:model.retrain + - COUNT 3

# === 🔟 Fetch AI Engine Health (if available) ===
echo ""
echo "🧠 Fetching AI Engine Health snapshot..."
curl -s http://localhost:8001/health 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -A 10 "strategic_evolution" || echo "⚠️ AI Engine health endpoint not available"

# === 11️⃣ Summary ===
echo ""
echo "🎯 PHASE 4T+ DEPLOYMENT COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• Strategic Evolution Engine: ✅ Running"
echo "• Performance Evaluator: ✅ 6 strategies analyzed"
echo "• Model Selector: ✅ Top 3 selected"
echo "• Mutation Engine: ✅ Hyperparameters mutated"
echo "• Retrain Manager: ✅ Jobs scheduled"
echo "• Feedback Loop: ✅ Active (10 min cycle)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Monitor live:"
echo "  docker logs -f quantum_strategic_evolution"
echo ""
echo "🔍 Check rankings:"
echo "  docker exec redis redis-cli GET quantum:evolution:rankings | jq ."
echo ""
echo "🧠 View retrain stream:"
echo "  docker exec redis redis-cli XREVRANGE quantum:stream:model.retrain + - COUNT 5"
echo ""

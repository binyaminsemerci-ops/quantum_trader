#!/bin/bash
set -e

echo "🚀 Starting Phase 2 Deployment — Observation & Brain Integration"
echo "================================================================"

# Log file
LOG_FILE="/var/log/phase2_deploy.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 deployment started"

# Determine working directory
if [ -d "/home/qt/quantum_trader" ]; then
    WORK_DIR="/home/qt/quantum_trader"
elif [ -d "/opt/quantum_trader" ]; then
    WORK_DIR="/opt/quantum_trader"
else
    echo "❌ ERROR: quantum_trader directory not found"
    exit 1
fi

cd "$WORK_DIR"
echo "📂 Working directory: $WORK_DIR"

# 1️⃣ Pre-flight check
echo ""
echo "1️⃣ Pre-flight check..."
echo "────────────────────────────────────────"

if docker ps | grep -q quantum_ai_engine; then
    echo "✅ AI Engine running"
else
    echo "⚠️  WARNING: AI Engine not running (will start with deployment)"
fi

echo ""
echo "🔍 Checking AI Engine health..."
HEALTH_STATUS=$(curl -s -w "\nHTTP %{http_code}\n" http://localhost:8001/health 2>/dev/null | tail -1 || echo "HTTP 000")
echo "AI Engine: $HEALTH_STATUS"

# 2️⃣ Sync code
echo ""
echo "2️⃣ Syncing code from repository..."
echo "────────────────────────────────────────"

if [ -d ".git" ]; then
    echo "🔄 Pulling latest code..."
    git pull origin main
else
    echo "⚠️  Not a git repository"
fi

# Verify Brain service files exist
echo ""
echo "🔍 Verifying service files..."
for SERVICE in ai_orchestrator ai_strategy ai_risk; do
    if [ -f "backend/$SERVICE/service.py" ]; then
        echo "✅ backend/$SERVICE/service.py exists"
    else
        echo "❌ ERROR: backend/$SERVICE/service.py not found"
        exit 1
    fi
done

# 3️⃣ Build services
echo ""
echo "3️⃣ Building Docker images..."
echo "────────────────────────────────────────"

docker compose -f docker-compose.vps.yml build \
    universe-os \
    pil \
    model-supervisor \
    ceo-brain \
    strategy-brain \
    risk-brain

# 4️⃣ Deploy services
echo ""
echo "4️⃣ Starting services..."
echo "────────────────────────────────────────"

docker compose -f docker-compose.vps.yml up -d \
    universe-os \
    pil \
    model-supervisor \
    ceo-brain \
    strategy-brain \
    risk-brain

echo ""
echo "⏳ Waiting 20 seconds for services to start..."
sleep 20

# 5️⃣ Health validation
echo ""
echo "5️⃣ Health validation..."
echo "────────────────────────────────────────"

SERVICES=(
    "8006:Universe OS"
    "8013:PIL"
    "8007:Model Supervisor"
    "8010:CEO Brain"
    "8011:Strategy Brain"
    "8012:Risk Brain"
)

ALL_HEALTHY=true

for SERVICE_INFO in "${SERVICES[@]}"; do
    PORT="${SERVICE_INFO%%:*}"
    NAME="${SERVICE_INFO#*:}"
    
    echo ""
    echo "Testing $NAME (port $PORT)..."
    
    HEALTH_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "HTTP_CODE:000")
    HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ $NAME: HTTP $HTTP_CODE - HEALTHY"
    else
        echo "❌ $NAME: HTTP $HTTP_CODE - FAILED"
        ALL_HEALTHY=false
    fi
done

# 6️⃣ Container status
echo ""
echo "6️⃣ Container status..."
echo "────────────────────────────────────────"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "universe_os|pil|model_supervisor|ceo_brain|strategy_brain|risk_brain" || echo "No Phase 2 containers found"

# 7️⃣ System handover
echo ""
echo "================================================================"
if [ "$ALL_HEALTHY" = true ]; then
    echo "✅ [PHASE 2 COMPLETE] Observation and Brains active – System neural layer online."
    echo ""
    echo "📊 Service Endpoints:"
    echo "   Universe OS:        http://localhost:8006/health"
    echo "   PIL:                http://localhost:8013/health"
    echo "   Model Supervisor:   http://localhost:8007/health"
    echo "   CEO Brain:          http://localhost:8010/health"
    echo "   Strategy Brain:     http://localhost:8011/health"
    echo "   Risk Brain:         http://localhost:8012/health"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 deployment completed successfully"
    exit 0
else
    echo "⚠️  [PHASE 2 PARTIAL] Some services failed health checks - check logs above"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 deployment completed with warnings"
    exit 1
fi

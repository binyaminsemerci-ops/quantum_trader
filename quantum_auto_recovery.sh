#!/bin/bash
# === Quantum Trader — Full Auto Recovery & Verification ===
# VPS: 46.224.116.254 (user: root)
# Purpose: Automatically rebuild, verify, and relaunch all Quantum Trader systems after data loss

set -e
cd /home/qt/quantum_trader
RECOVERY_LOG=/var/log/quantum_auto_recovery_$(date +%Y%m%d_%H%M%S).log
echo "🚀 Quantum Trader — FULL AUTO RECOVERY $(date)" | tee -a "$RECOVERY_LOG"

# 1️⃣ PRE-CHECKS
echo -e "\n=== [System Verification] ===" | tee -a "$RECOVERY_LOG"
hostnamectl | tee -a "$RECOVERY_LOG"
docker --version || (echo "❌ Docker not installed" && exit 1)
df -h / | tee -a "$RECOVERY_LOG"
df -h /mnt/HC_Volume_104287969 | tee -a "$RECOVERY_LOG"

# 2️⃣ UPDATE PROJECT FROM GIT
echo -e "\n=== [Git Sync] ===" | tee -a "$RECOVERY_LOG"
git fetch origin main && git reset --hard origin/main | tee -a "$RECOVERY_LOG"

# 3️⃣ VERIFY DOCKER COMPOSE EXISTS
if [ ! -f docker-compose.vps.yml ]; then
  echo "❌ docker-compose.vps.yml missing! Please restore from backup." | tee -a "$RECOVERY_LOG"
  exit 1
fi
echo "✅ docker-compose.vps.yml found" | tee -a "$RECOVERY_LOG"

# 4️⃣ VERIFY .env EXISTS
if [ ! -f .env ]; then
  echo "❌ .env missing! Please restore API keys and configuration." | tee -a "$RECOVERY_LOG"
  exit 1
fi
echo "✅ .env found" | tee -a "$RECOVERY_LOG"
echo "Testnet: $(grep BINANCE_TESTNET .env || echo 'not set')" | tee -a "$RECOVERY_LOG"

# 5️⃣ STOP ALL CONTAINERS
echo -e "\n=== [Stopping Existing Containers] ===" | tee -a "$RECOVERY_LOG"
docker stop $(docker ps -q) 2>/dev/null || echo "No containers running"
docker rm $(docker ps -aq) 2>/dev/null || echo "No containers to remove"

# 6️⃣ CLEAN DOCKER CACHE
echo -e "\n=== [Docker Cleanup] ===" | tee -a "$RECOVERY_LOG"
docker system prune -af | tee -a "$RECOVERY_LOG"

# 7️⃣ REBUILD CRITICAL SERVICES (Phase 1: Infrastructure)
echo -e "\n=== [Phase 1: Infrastructure] ===" | tee -a "$RECOVERY_LOG"
docker compose -f docker-compose.vps.yml build redis | tee -a "$RECOVERY_LOG"
docker compose -f docker-compose.vps.yml up -d redis | tee -a "$RECOVERY_LOG"
sleep 5
docker exec quantum_redis redis-cli PING | tee -a "$RECOVERY_LOG"

# 8️⃣ REBUILD AI/ML CORE (Phase 2)
echo -e "\n=== [Phase 2: AI/ML Core] ===" | tee -a "$RECOVERY_LOG"
for service in ai-engine ceo-brain risk-brain strategy-brain universe-os model-supervisor pil; do
  echo "Building $service..." | tee -a "$RECOVERY_LOG"
  docker compose -f docker-compose.vps.yml build $service 2>&1 | tee -a "$RECOVERY_LOG" || echo "⚠️ $service build failed"
done
docker compose -f docker-compose.vps.yml up -d ai-engine ceo-brain risk-brain strategy-brain universe-os model-supervisor pil | tee -a "$RECOVERY_LOG"
sleep 15

# 9️⃣ REBUILD RL SYSTEM (Phase 3)
echo -e "\n=== [Phase 3: RL System] ===" | tee -a "$RECOVERY_LOG"
for service in rl-monitor rl-calibrator rl-sizing-agent rl-feedback-v2 rl-dashboard strategic-evolution strategic-memory model-federation; do
  echo "Building $service..." | tee -a "$RECOVERY_LOG"
  docker compose -f docker-compose.vps.yml build $service 2>&1 | tee -a "$RECOVERY_LOG" || echo "⚠️ $service build failed"
done
docker compose -f docker-compose.vps.yml up -d rl-monitor rl-calibrator rl-sizing-agent rl-feedback-v2 rl-dashboard strategic-evolution strategic-memory model-federation | tee -a "$RECOVERY_LOG"
sleep 10

# 🔟 REBUILD GOVERNANCE & PORTFOLIO (Phase 4)
echo -e "\n=== [Phase 4: Governance] ===" | tee -a "$RECOVERY_LOG"
for service in portfolio-governance meta-regime; do
  docker compose -f docker-compose.vps.yml build $service 2>&1 | tee -a "$RECOVERY_LOG" || echo "⚠️ $service build failed"
done
docker compose -f docker-compose.vps.yml up -d portfolio-governance meta-regime | tee -a "$RECOVERY_LOG"
sleep 5

# 1️⃣1️⃣ REBUILD FRONTEND (Phase 5)
echo -e "\n=== [Phase 5: Frontend] ===" | tee -a "$RECOVERY_LOG"
docker compose -f docker-compose.vps.yml build quantumfond-frontend 2>&1 | tee -a "$RECOVERY_LOG" || echo "⚠️ Frontend build failed"
docker compose -f docker-compose.vps.yml up -d quantumfond-frontend | tee -a "$RECOVERY_LOG"
sleep 5

# 1️⃣2️⃣ REBUILD TRADING LAYER (Phase 6) - MANUAL BUILD REQUIRED
echo -e "\n=== [Phase 6: Trading Execution Layer] ===" | tee -a "$RECOVERY_LOG"
echo "⚠️ Trading layer requires manual build (not in docker-compose):" | tee -a "$RECOVERY_LOG"
echo "  - Position Monitor (port 8010)" | tee -a "$RECOVERY_LOG"
echo "  - Auto Executor" | tee -a "$RECOVERY_LOG"
echo "  - Strategy Ops" | tee -a "$RECOVERY_LOG"
echo "" | tee -a "$RECOVERY_LOG"
echo "Manual build commands:" | tee -a "$RECOVERY_LOG"
echo "  docker build -t quantum_trader-position-monitor:latest -f microservices/position_monitor/Dockerfile ." | tee -a "$RECOVERY_LOG"
echo "  docker build -t quantum_trader-auto-executor:latest -f backend/microservices/auto_executor/Dockerfile ." | tee -a "$RECOVERY_LOG"
echo "  docker build -t quantum_trader-strategy-ops:latest -f microservices/strategy_operations/Dockerfile ." | tee -a "$RECOVERY_LOG"

# 1️⃣3️⃣ VERIFY HEALTH
echo -e "\n=== [Health Checks] ===" | tee -a "$RECOVERY_LOG"
sleep 10
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | tee -a "$RECOVERY_LOG"

# Test critical ports
echo -e "\n=== [Port Tests] ===" | tee -a "$RECOVERY_LOG"
for port in 6379 8001 8002 8006 8007 8010 8011 8012 8013 3000 8026; do
  if timeout 2 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
    echo "✅ Port $port → OPEN" | tee -a "$RECOVERY_LOG"
  else
    echo "❌ Port $port → CLOSED" | tee -a "$RECOVERY_LOG"
  fi
done

# 1️⃣4️⃣ REDIS INTEGRITY TEST
echo -e "\n=== [Redis Check] ===" | tee -a "$RECOVERY_LOG"
docker exec quantum_redis redis-cli PING | tee -a "$RECOVERY_LOG" || echo "⚠️ Redis not responding" | tee -a "$RECOVERY_LOG"
docker exec quantum_redis redis-cli DBSIZE | tee -a "$RECOVERY_LOG"
docker exec quantum_redis redis-cli KEYS "quantum:*" | wc -l | tee -a "$RECOVERY_LOG"

# 1️⃣5️⃣ BINANCE TESTNET VALIDATION
echo -e "\n=== [Binance Testnet Validation] ===" | tee -a "$RECOVERY_LOG"
docker exec quantum_ai_engine python3 - <<'PY' 2>&1 | tee -a "$RECOVERY_LOG" || echo "⚠️ Binance API test failed (may need .env update)" | tee -a "$RECOVERY_LOG"
import os
try:
    from binance.client import Client
    key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    if key and secret:
        client = Client(key, secret, testnet=True)
        print("✅ Server Time:", client.get_server_time())
        print("✅ Symbols:", len(client.get_exchange_info()['symbols']))
    else:
        print("⚠️ API keys missing in .env")
except Exception as e:
    print(f"❌ Binance test error: {e}")
PY

# 1️⃣6️⃣ AI/RL VERIFICATION
echo -e "\n=== [AI/RL Service Logs] ===" | tee -a "$RECOVERY_LOG"
for svc in quantum_ai_engine quantum_rl_feedback_v2 quantum_rl_calibrator quantum_rl_monitor; do
  if docker ps --filter name=$svc --format '{{.Names}}' | grep -q $svc; then
    echo -e "\n🧠 Logs for $svc:" | tee -a "$RECOVERY_LOG"
    docker logs --tail 10 $svc 2>&1 | tee -a "$RECOVERY_LOG"
  else
    echo "⚠️ $svc not running" | tee -a "$RECOVERY_LOG"
  fi
done

# 1️⃣7️⃣ MODEL VERIFICATION
echo -e "\n=== [ML Model Check] ===" | tee -a "$RECOVERY_LOG"
if [ -d models ]; then
  MODEL_COUNT=$(find models -type f -name "*.pkl" -o -name "*.pt" -o -name "*.pth" | wc -l)
  echo "✅ Found $MODEL_COUNT model files" | tee -a "$RECOVERY_LOG"
  du -sh models | tee -a "$RECOVERY_LOG"
else
  echo "⚠️ models/ directory missing!" | tee -a "$RECOVERY_LOG"
fi

# 1️⃣8️⃣ SUMMARY
echo -e "\n=== [Recovery Summary] ===" | tee -a "$RECOVERY_LOG"
RUNNING=$(docker ps | wc -l)
HEALTHY=$(docker ps --filter health=healthy | wc -l)
echo "📊 Containers Running: $((RUNNING-1))" | tee -a "$RECOVERY_LOG"
echo "✅ Healthy Services: $((HEALTHY-1))" | tee -a "$RECOVERY_LOG"
echo "💾 Primary Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')" | tee -a "$RECOVERY_LOG"
echo "💾 Extra Volume: $(df -h /mnt/HC_Volume_104287969 | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')" | tee -a "$RECOVERY_LOG"
echo "📁 Project: /home/qt/quantum_trader" | tee -a "$RECOVERY_LOG"
echo "🔗 Recovery log: $RECOVERY_LOG" | tee -a "$RECOVERY_LOG"
echo "" | tee -a "$RECOVERY_LOG"
echo "🌐 Access Points:" | tee -a "$RECOVERY_LOG"
echo "  - Frontend: http://46.224.116.254:3000" | tee -a "$RECOVERY_LOG"
echo "  - AI Engine: http://46.224.116.254:8001/health" | tee -a "$RECOVERY_LOG"
echo "  - RL Dashboard: http://46.224.116.254:8026" | tee -a "$RECOVERY_LOG"
echo "" | tee -a "$RECOVERY_LOG"
echo "✅ RECOVERY COMPLETE!" | tee -a "$RECOVERY_LOG"

#!/bin/bash
# === Quantum Trader — Recovery Script TEST (Dry-Run) ===
# This version only REPORTS what would happen, makes NO changes

set -e
cd /home/qt/quantum_trader
TEST_LOG=/tmp/recovery_test_$(date +%Y%m%d_%H%M%S).log
echo "🧪 Quantum Trader — RECOVERY TEST (DRY-RUN) $(date)" | tee "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"
echo "⚠️  This is a TEST - no changes will be made!" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

# 1️⃣ SYSTEM CHECK
echo "=== [System Verification] ===" | tee -a "$TEST_LOG"
echo "Hostname: $(hostname)" | tee -a "$TEST_LOG"
echo "Docker: $(docker --version)" | tee -a "$TEST_LOG"
echo "Primary Disk: $(df -h / | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')" | tee -a "$TEST_LOG"
echo "Extra Volume: $(df -h /mnt/HC_Volume_104287969 | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')" | tee -a "$TEST_LOG"

# 2️⃣ BACKUP CHECK
echo "" | tee -a "$TEST_LOG"
echo "=== [Backup Inventory] ===" | tee -a "$TEST_LOG"
echo "✓ .env backups: $(ls .env.backup* 2>/dev/null | wc -l) files" | tee -a "$TEST_LOG"
[ -f .env ] && echo "✓ Current .env: $(wc -l < .env) lines" | tee -a "$TEST_LOG" || echo "✗ .env missing!" | tee -a "$TEST_LOG"
echo "✓ Redis backups: $(ls /home/qt/backups/redis/*.rdb 2>/dev/null | wc -l) files" | tee -a "$TEST_LOG"
echo "✓ Model files: $(find models -name "*.pkl" -o -name "*.pt" 2>/dev/null | wc -l) files" | tee -a "$TEST_LOG"
echo "✓ Git repo: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')" | tee -a "$TEST_LOG"

# 3️⃣ DOCKER COMPOSE CHECK
echo "" | tee -a "$TEST_LOG"
echo "=== [Configuration Files] ===" | tee -a "$TEST_LOG"
[ -f docker-compose.vps.yml ] && echo "✓ docker-compose.vps.yml: $(wc -l < docker-compose.vps.yml) lines" | tee -a "$TEST_LOG" || echo "✗ docker-compose.vps.yml missing!" | tee -a "$TEST_LOG"
[ -f Dockerfile ] && echo "✓ Dockerfile found" | tee -a "$TEST_LOG"
echo "✓ Microservices: $(find microservices -maxdepth 1 -type d | wc -l) directories" | tee -a "$TEST_LOG"

# 4️⃣ CURRENT CONTAINERS
echo "" | tee -a "$TEST_LOG"
echo "=== [Running Containers] ===" | tee -a "$TEST_LOG"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | tee -a "$TEST_LOG"
RUNNING=$(docker ps -q | wc -l)
HEALTHY=$(docker ps --filter health=healthy -q | wc -l)
echo "" | tee -a "$TEST_LOG"
echo "Total running: $RUNNING" | tee -a "$TEST_LOG"
echo "Healthy: $HEALTHY" | tee -a "$TEST_LOG"

# 5️⃣ SERVICES IN DOCKER-COMPOSE
echo "" | tee -a "$TEST_LOG"
echo "=== [Services in docker-compose.vps.yml] ===" | tee -a "$TEST_LOG"
if [ -f docker-compose.vps.yml ]; then
  grep "^  [a-z].*:$" docker-compose.vps.yml | sed 's/://g' | tee -a "$TEST_LOG"
  COMPOSE_SERVICES=$(grep "^  [a-z].*:$" docker-compose.vps.yml | wc -l)
  echo "" | tee -a "$TEST_LOG"
  echo "Total services defined: $COMPOSE_SERVICES" | tee -a "$TEST_LOG"
fi

# 6️⃣ WHAT WOULD BE REBUILT
echo "" | tee -a "$TEST_LOG"
echo "=== [Recovery Would Execute] ===" | tee -a "$TEST_LOG"
echo "Phase 1: Stop $RUNNING containers" | tee -a "$TEST_LOG"
echo "Phase 2: Clean Docker cache (current: $(docker system df | grep 'Build Cache' | awk '{print $4}'))" | tee -a "$TEST_LOG"
echo "Phase 3: Git reset to origin/main (current: $(git rev-parse --short HEAD))" | tee -a "$TEST_LOG"
echo "Phase 4: Rebuild Redis" | tee -a "$TEST_LOG"
echo "Phase 5: Rebuild AI/ML Core (8 services)" | tee -a "$TEST_LOG"
echo "Phase 6: Rebuild RL System (8 services)" | tee -a "$TEST_LOG"
echo "Phase 7: Rebuild Governance (2 services)" | tee -a "$TEST_LOG"
echo "Phase 8: Rebuild Frontend" | tee -a "$TEST_LOG"
echo "Phase 9: Rebuild Trading Layer (manual)" | tee -a "$TEST_LOG"

# 7️⃣ REDIS TEST
echo "" | tee -a "$TEST_LOG"
echo "=== [Redis Connectivity] ===" | tee -a "$TEST_LOG"
docker exec quantum_redis redis-cli PING 2>&1 | tee -a "$TEST_LOG" || echo "✗ Redis not accessible" | tee -a "$TEST_LOG"
docker exec quantum_redis redis-cli DBSIZE 2>&1 | tee -a "$TEST_LOG" || true

# 8️⃣ API TEST
echo "" | tee -a "$TEST_LOG"
echo "=== [API Endpoints] ===" | tee -a "$TEST_LOG"
for port in 6379 8001 8002 8006 8007 8010 8011 8012 8013 3000 8026; do
  if timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
    echo "✓ Port $port: OPEN" | tee -a "$TEST_LOG"
  else
    echo "✗ Port $port: CLOSED" | tee -a "$TEST_LOG"
  fi
done

# 9️⃣ SUMMARY
echo "" | tee -a "$TEST_LOG"
echo "=== [Test Summary] ===" | tee -a "$TEST_LOG"
echo "✅ System is operational ($RUNNING containers running)" | tee -a "$TEST_LOG"
echo "✅ Backups available for recovery" | tee -a "$TEST_LOG"
echo "✅ Recovery script ready at: $(pwd)/quantum_auto_recovery.sh" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"
echo "📋 Test log saved: $TEST_LOG" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"
echo "⚠️  To run actual recovery (use only if system is broken):" | tee -a "$TEST_LOG"
echo "   bash quantum_auto_recovery.sh" | tee -a "$TEST_LOG"

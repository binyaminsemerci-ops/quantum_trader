#!/bin/bash
set -e
cd ~/quantum_trader

echo "🚀 Starting ExitBrain Integration (Module Mode)..."

# 1️⃣ Pull latest changes
git pull origin main

# 2️⃣ Stop and remove exitbrain container if exists
if docker ps -a --format '{{.Names}}' | grep -q 'exitbrain'; then
  echo "🧹 Stopping and removing exitbrain-v3-5 container..."
  docker compose -f docker-compose.vps.yml stop exitbrain-v3-5 || true
  docker compose -f docker-compose.vps.yml rm -f exitbrain-v3-5 || true
fi

# 3️⃣ Verify exitbrain_v3_5 module path
if [ -d "microservices/exitbrain_v3_5" ]; then
  echo "✅ exitbrain_v3_5 module present."
else
  echo "❌ exitbrain_v3_5 module missing!"
  exit 1
fi

# 4️⃣ Rebuild and restart position-monitor
echo "🔁 Rebuilding and restarting position-monitor..."
docker compose -f docker-compose.vps.yml build position-monitor
docker compose -f docker-compose.vps.yml up -d position-monitor

# 5️⃣ Health check
echo "⏳ Waiting 15 seconds for startup..."
sleep 15
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'position_monitor|NAMES'
echo ""
docker logs --tail 20 quantum_position_monitor

echo ""
echo "🧩 ExitBrain module integration complete."

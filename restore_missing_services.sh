#!/bin/bash
# === Quantum Trader VPS – Restore Missing Microservices ===
set -e
cd /home/qt/quantum_trader

echo "🧩 Restoring missing services: exitbrain_v3_5, position_monitor"

# 1️⃣ Ensure directories exist
for dir in microservices/exitbrain_v3_5 microservices/position_monitor; do
  [ -d "$dir" ] && echo "✅ $dir found" || { echo "❌ $dir missing – please sync from dev"; exit 1; }
done

# 2️⃣ Append services to docker-compose.vps.yml if absent
grep -q "exitbrain-v3-5" docker-compose.vps.yml || cat <<'YAML' >> docker-compose.vps.yml

exitbrain-v3-5:
  build: ./microservices/exitbrain_v3_5
  container_name: quantum_exitbrain_v3_5
  restart: always
  depends_on:
    - redis
    - ai-engine
  networks:
    - quantum_trader
  ports:
    - "8015:8015"

position-monitor:
  build: ./microservices/position_monitor
  container_name: quantum_position_monitor
  restart: always
  depends_on:
    - ai-engine
    - redis
  networks:
    - quantum_trader
  ports:
    - "8016:8016"
YAML

# 3️⃣ Build and start only the missing ones
docker compose -f docker-compose.vps.yml build exitbrain-v3-5 position-monitor
docker compose -f docker-compose.vps.yml up -d exitbrain-v3-5 position-monitor

# 4️⃣ Health verification
sleep 8
for port in 8015 8016; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health || true)
  echo "Port $port → HTTP $code"
done

echo "✅ Missing services restored successfully."

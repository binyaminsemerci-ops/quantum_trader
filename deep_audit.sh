#!/bin/bash
# === Quantum Deep Integrity Audit ===
LOGFILE="/var/log/quantum_deep_audit.log"
exec > >(tee -a "$LOGFILE") 2>&1
set -e

echo "🚀 Starting deep audit $(date)"

## 1.  Container and network state
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
docker network ls | grep quantum_trader || echo "⚠️ Network quantum_trader missing"

## 2.  Source-tree verification (should exist from yesterday's build)
find ~/quantum_trader -type d \( -name "microservices" -o -name "backend" \) -maxdepth 2 | sort

## 3.  Health endpoints
for port in {8001..8015}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health || true)
  printf "Port %-5s → HTTP %s\n" "$port" "$code"
done

## 4.  Redis core keys & pub/sub
echo "Redis namespaces:"
docker exec redis redis-cli KEYS "quantum:*" | head -20
echo "Checking live channels:"
docker exec redis redis-cli PUBSUB CHANNELS

## 5.  Internal API communication checks
# Verify that brains and engines can talk through HTTP
for svc in ai_engine strategy_brain risk_brain ceo_brain universe_os model_supervisor; do
  host=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' quantum_${svc} 2>/dev/null || true)
  [ -n "$host" ] && curl -s -o /dev/null -w "$svc → %{http_code}\n" http://$host:8000/health || echo "$svc unreachable"
done

## 6.  Log integrity
tail -n 10 /var/log/*_brain*.log 2>/dev/null || echo "No brain logs found"
tail -n 10 /var/log/model_federation*.log 2>/dev/null || true

## 7.  Git & config sync
cd ~/quantum_trader && git fetch origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "⚠️ Local repo not synced with origin/main"
else
  echo "✅ Git repo up-to-date"
fi

## 8.  Summarize critical modules
for m in exit_brain_v3 model_federation position_monitor order_manager ai_engine universe_os; do
  docker ps --format '{{.Names}}' | grep -q $m && echo "✅ $m active" || echo "❌ $m missing"
done

echo "🧩 Deep audit finished — full log in $LOGFILE"

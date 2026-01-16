#!/bin/bash
# === Quantum Deep Integrity Audit ===
LOGFILE="/var/log/quantum_deep_audit.log"
exec > >(tee -a "$LOGFILE") 2>&1
set -e

echo "🚀 Starting deep audit $(date)"

## 1.  Service and system state
systemctl list-units 'quantum-*.service' --no-pager --no-legend | head -20
echo "Network: systemd networking (no docker network needed)"

## 2.  Source-tree verification (should exist from yesterday's build)
find /home/qt/quantum_trader -type d \( -name "microservices" -o -name "backend" \) -maxdepth 2 | sort

## 3.  Health endpoints
for port in {8001..8015}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health || true)
  printf "Port %-5s → HTTP %s\n" "$port" "$code"
done

## 4.  Redis core keys & pub/sub
echo "Redis namespaces:"
redis-cli KEYS "quantum:*" | head -20
echo "Checking live channels:"
redis-cli PUBSUB CHANNELS

## 5.  Service health checks
# Verify that brains and engines are responding via HTTP
for svc in ai-engine strategy-brain risk-brain ceo-brain universe-os model-supervisor; do
  curl -s -o /dev/null -w "quantum-$svc.service → %{http_code}\n" http://localhost:8000/health || echo "$svc unreachable"
done

## 6.  Log integrity
tail -n 10 /var/log/*_brain*.log 2>/dev/null || echo "No brain logs found"
tail -n 10 /var/log/model_federation*.log 2>/dev/null || true

## 7.  Git & config sync
cd /home/qt/quantum_trader && git fetch origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "⚠️ Local repo not synced with origin/main"
else
  echo "✅ Git repo up-to-date"
fi

## 8.  Summarize critical modules
for m in exit-brain-v3 model-federation position-monitor order-manager ai-engine universe-os; do
  systemctl is-active quantum-$m.service >/dev/null 2>&1 && echo "✅ $m active" || echo "❌ $m missing"
done

echo "🧩 Deep audit finished — full log in $LOGFILE"

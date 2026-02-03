#!/bin/bash
set -e
echo "🧠 Quantum Trader RL Startup $(date)"
cd ~/quantum_trader

# Function to wait for a service
wait_for() {
  local name=$1
  local cmd=$2
  echo "⏳ Waiting for $name..."
  for i in {1..20}; do
    if eval "$cmd" >/dev/null 2>&1; then
      echo "✅ $name is ready"
      return 0
    fi
    sleep 3
  done
  echo "❌ $name failed to start in time"
  exit 1
}

echo "🔧 Checking Docker..."
docker info >/dev/null 2>&1 || (echo "Docker not running" && exit 1)

echo "📡 Starting Redis..."
docker compose -f docker-compose.vps.yml up -d redis --no-build
wait_for "Redis" "docker exec quantum_redis redis-cli PING | grep -q PONG"

echo "🧩 Starting StrategyOps..."
docker compose -f docker-compose.vps.yml up -d strategy-ops --no-build
wait_for "StrategyOps" "docker logs quantum_strategy_ops 2>&1 | grep -q 'StrategyOps active'"

echo "🧬 Starting RL Feedback Bridge..."
docker compose -f docker-compose.vps.yml up -d rl-feedback-v2 --no-build
wait_for "RL Feedback Bridge" "docker logs quantum_rl_feedback_v2 2>&1 | grep -q 'RL Feedback Bridge v2 running'"

echo "📊 Starting RL Dashboard..."
docker compose -f docker-compose.vps.yml up -d rl-dashboard --no-build
wait_for "RL Dashboard" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8027 | grep -q 200"

echo "✅ All Quantum RL Services Started Successfully!"
echo "🧠 Redis + RL Feedback + StrategyOps + Dashboard running."
docker ps --format "table {{.Names}}\t{{.Status}}"

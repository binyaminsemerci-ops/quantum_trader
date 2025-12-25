#!/bin/bash
# Quantum Trader Learning Systems Audit
# Target: VPS 46.224.116.254

set -e

echo "🧩 Starting Quantum Trader Learning Systems Audit — $(date)"

# 1️⃣ Container Health
echo -e "\n=== [Container Health] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep quantum'

echo -e "\n=== [Health Endpoints] ==="
for port in 8001 8006 8008 8010 8011 8012 8016; do
  code=$(ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "curl -s -o /dev/null -w '%{http_code}' http://localhost:$port/health" 2>/dev/null || echo "000")
  echo "Port $port → HTTP $code"
done

# 2️⃣ Supervised Learning
echo -e "\n=== [Supervised Models] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_ai_engine ls /app/models 2>&1 | head -10'
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli KEYS "quantum:model:*:accuracy"'

# 3️⃣ Trust Memory
echo -e "\n=== [Trust Memory / Meta-Learning] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli KEYS "quantum:trust:*"'

# 4️⃣ Reinforcement Learning
echo -e "\n=== [Reinforcement Learning] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli KEYS "quantum:reward:*" 2>&1 || echo "(no reward keys)"'
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli KEYS "quantum:rl:*" 2>&1 || echo "(no rl keys)"'

# 5️⃣ Model Federation
echo -e "\n=== [Model Federation Ensemble] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli GET quantum:consensus:signal 2>&1'

# 6️⃣ Context Awareness
echo -e "\n=== [Context / Market Regime] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli KEYS "quantum:context:*"'
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli KEYS "quantum:regime:*"'

# 7️⃣ Redis Memory
echo -e "\n=== [Redis Memory] ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli INFO memory | grep used_memory_human'
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker exec quantum_redis redis-cli DBSIZE'

# 8️⃣ Summary
echo -e "\n=== [Audit Summary] ==="
echo "✅  Supervised models: Check /app/models directory"
echo "✅  Meta-learning: Check trust:* keys"
echo "✅  Reinforcement: Check reward:* or rl:* keys"
echo "✅  Federation: Check consensus:signal"
echo "✅  Context: Check context:* or regime:* keys"

echo -e "\n🧠 Learning Systems Audit completed — $(date)"

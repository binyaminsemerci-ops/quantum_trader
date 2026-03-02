#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

$SSH 'bash -s' << 'ENDSSH'
echo "========================================="
echo "QUANTUM TRADER - FULL SYSTEM AUDIT"
echo "Date: $(date -u)"
echo "========================================="

echo ""
echo "=== 1. RUNNING SERVICES ==="
for s in quantum-ai-engine quantum-backend quantum-market-publisher quantum-rl-trainer quantum-rl-monitor; do
  status=$(systemctl is-active $s 2>/dev/null || echo "NOT_FOUND")
  pid=$(systemctl show $s -p MainPID --value 2>/dev/null)
  echo "  $s: $status (PID=$pid)"
done

echo ""
echo "=== 2. REDIS ==="
redis-cli ping 2>/dev/null || echo "REDIS NOT RESPONDING"
redis-cli info server 2>/dev/null | grep -E 'redis_version|uptime_in_days|connected_clients|used_memory_human'

echo ""
echo "=== 3. PROCESS LIST ==="
ps aux --sort=-%cpu | grep -E '(python|uvicorn|redis)' | grep -v grep | awk '{printf "  %-8s %-5s %-5s %s\n", $1, $3, $4, $11}'

echo ""
echo "=== 4. DISK & MEMORY ==="
df -h / 2>/dev/null | tail -1
free -h | grep Mem

echo ""
echo "=== 5. GIT LOG (VPS) ==="
cd /opt/quantum && git log --oneline -10

echo ""
echo "=== 6. AI ENGINE LAST LOGS (errors/warnings) ==="
journalctl -u quantum-ai-engine -n 200 --no-pager -q 2>/dev/null | grep -E '"level": "(ERROR|WARNING)"' | tail -15

echo ""
echo "=== 7. ENSEMBLE STATUS ==="
journalctl -u quantum-ai-engine -n 200 --no-pager -q 2>/dev/null | grep 'QSC.*ACTIVE' | tail -5

echo ""
echo "=== 8. MODELS DIRECTORY ==="
ls -lh /opt/quantum/ai_engine/models/*.pkl 2>/dev/null | grep -v scaler_v | awk '{print $5, $6, $7, $8, $9}' | sort -k3,4

echo ""
echo "=== 9. BACKEND LAST ERRORS ==="
journalctl -u quantum-backend -n 100 --no-pager -q 2>/dev/null | grep -E '"level": "(ERROR|WARNING)"' | tail -10

echo ""
echo "=== 10. OPEN POSITIONS (Redis) ==="
redis-cli keys 'quantum:position:*' 2>/dev/null | head -20
echo "Total position keys: $(redis-cli keys 'quantum:position:*' 2>/dev/null | wc -l)"

echo ""
echo "=== 11. CIRCUIT BREAKERS / RISK STATE (Redis) ==="
for key in quantum:circuit_breaker quantum:emergency_stop quantum:drawdown quantum:equity_usd quantum:max_leverage; do
  val=$(redis-cli get "$key" 2>/dev/null)
  echo "  $key = $val"
done

echo ""
echo "=== 12. KELLY / LEVERAGE KEYS ==="
redis-cli keys 'quantum:kelly*' 2>/dev/null
redis-cli keys 'quantum:leverage*' 2>/dev/null

echo ""
echo "=== 13. BACKEND SERVICE FILE (critical config) ==="
grep -E '(AI_MAX_LEVERAGE|EQUITY|KELLY|MAX_POSITION|CIRCUIT|DRAWDOWN|ENVIRONMENT)' /etc/systemd/system/quantum-backend.service 2>/dev/null | head -20

echo ""
echo "=== 14. AI ENGINE SERVICE FILE ==="
grep -E '(AI_MAX_LEVERAGE|EQUITY|KELLY|MAX_POSITION|CIRCUIT|DRAWDOWN|ENVIRONMENT|FEATURES_V6)' /etc/systemd/system/quantum-ai-engine.service 2>/dev/null | head -20

echo ""
echo "=== 15. RECENT TRADES (Redis stream) ==="
redis-cli xlen quantum:stream:trade.closed 2>/dev/null
redis-cli xrevrange quantum:stream:trade.closed + - COUNT 3 2>/dev/null | grep -E '(symbol|action|pnl|timestamp)' | head -20

echo ""
echo "=== 16. RL TRAINER STATUS ==="
journalctl -u quantum-rl-trainer -n 30 --no-pager -q 2>/dev/null | tail -8

echo ""
echo "=== 17. PORT LISTEN ==="
ss -tlnp 2>/dev/null | grep -E '(8000|8001|8002|6379|5432|3000)' | awk '{print $4, $6}'

echo ""
echo "=== 18. UNIFIED_AGENTS.PY AGENT CLASSES ==="
grep -n 'class.*Agent.*BaseAgent' /opt/quantum/ai_engine/agents/unified_agents.py 2>/dev/null

echo ""
echo "=== 19. XGB + LGBM LOAD CONFIRMATION ==="
journalctl -u quantum-ai-engine -n 500 --no-pager -q 2>/dev/null | grep -E '(Loaded.*pkl|Loaded.*pth|XGB-Agent.*WARN|LGBM.*WARN)' | grep -v 'Dropping' | tail -15

echo ""
echo "=== 20. ACTIVE SLOTS / POSITION CONTROLLER ==="
for key in quantum:active_slots quantum:slot_count quantum:max_slots quantum:positions_count; do
  val=$(redis-cli get "$key" 2>/dev/null)
  echo "  $key = $val"
done

echo "========================================="
echo "AUDIT COMPLETE"
echo "========================================="
ENDSSH

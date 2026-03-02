#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

$SSH 'bash -s' << 'ENDSSH'

echo "=== SERVICE STATUS ==="
for s in quantum-ai-engine quantum-backend quantum-market-publisher quantum-rl-trainer quantum-rl-monitor; do
  status=$(systemctl is-active $s 2>/dev/null)
  pid=$(systemctl show $s -p MainPID --value 2>/dev/null)
  mem=$(cat /proc/$pid/status 2>/dev/null | grep VmRSS | awk '{print $2, $3}')
  echo "  $s: $status | PID=$pid | MEM=$mem"
done

echo ""
echo "=== GIT LOG ==="
cd /opt/quantum && git log --oneline -10

echo ""
echo "=== AI ENGINE ENSEMBLE (latest) ==="
journalctl -u quantum-ai-engine -n 500 --no-pager -q 2>/dev/null | grep -E '(Loaded.*pkl|Loaded.*pth|QSC.*ACTIVE)' | tail -15

echo ""
echo "=== AI ENGINE ERRORS (last hour, unique) ==="
journalctl -u quantum-ai-engine --since "1 hour ago" --no-pager -q 2>/dev/null | grep '"level": "ERROR"' | sed 's/.*"msg": "//' | sed 's/".*//' | sort | uniq -c | sort -rn | head -15

echo ""
echo "=== BACKEND ERRORS (last hour, unique) ==="
journalctl -u quantum-backend --since "1 hour ago" --no-pager -q 2>/dev/null | grep '"level": "ERROR"' | sed 's/.*"msg": "//' | sed 's/".*//' | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== SERVICE ENV VARS ==="
echo "-- quantum-backend --"
grep -E 'Environment' /etc/systemd/system/quantum-backend.service 2>/dev/null | head -10
echo "-- quantum-ai-engine --"
grep -E 'Environment' /etc/systemd/system/quantum-ai-engine.service 2>/dev/null | head -10

echo ""
echo "=== RISK / EQUITY REDIS KEYS ==="
for key in "quantum:circuit_breaker" "quantum:emergency_stop" "quantum:drawdown" "quantum:drawdown_pct" "quantum:equity_usd" "quantum:equity" "quantum:max_leverage" "quantum:daily_loss" "quantum:trade_count:today" "quantum:rl:state" "quantum:kelly:fraction"; do
  val=$(redis-cli get "$key" 2>/dev/null)
  [ -n "$val" ] && echo "  $key = $val"
done
echo "(empty keys not shown)"

echo ""
echo "=== OPEN POSITIONS (live, non-snapshot) ==="
redis-cli keys 'quantum:position:[A-Z1]*' 2>/dev/null | grep -v 'snapshot\|ledger' | sort | head -20
poscount=$(redis-cli keys 'quantum:position:[A-Z1]*' 2>/dev/null | grep -v 'snapshot\|ledger' | wc -l)
echo "Active positions: $poscount"

echo ""
echo "=== LAST 3 CLOSED TRADES ==="
redis-cli xrevrange quantum:stream:trade.closed + - COUNT 3 2>/dev/null | grep -E '^\S|symbol|pnl_percent|pnl_usd|side|timestamp' | head -30

echo ""
echo "=== RL TRAINER ==="
journalctl -u quantum-rl-trainer -n 20 --no-pager -q 2>/dev/null | tail -8

echo ""
echo "=== DISK ==="
echo "Old stale xgb_model_v* files: $(ls /opt/quantum/ai_engine/models/xgb_model_v*.pkl 2>/dev/null | wc -l)"
echo "Total model files: $(ls /opt/quantum/ai_engine/models/*.pkl 2>/dev/null | wc -l)"
du -sh /opt/quantum/ai_engine/models/ 2>/dev/null
df -h / 2>/dev/null | tail -1

echo ""
echo "=== PERMIT KEY COUNT ==="
redis-cli keys 'quantum:permit:*' 2>/dev/null | wc -l

echo ""
echo "=== ACTIVE SLOT / POSITION CONTROLLER KEYS ==="
for key in "quantum:active_slots" "quantum:slot_count" "quantum:max_slots" "quantum:positions_count" "quantum:position_limit"; do
  val=$(redis-cli get "$key" 2>/dev/null)
  echo "  $key = ${val:-<empty>}"
done

echo ""
echo "=== BACKEND SERVICE HEALTHY? ==="
curl -s --max-time 5 http://localhost:8000/health 2>/dev/null | head -c 500

ENDSSH

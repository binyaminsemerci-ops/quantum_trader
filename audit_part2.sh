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
echo "=== GIT LOG (VPS /opt/quantum) ==="
cd /opt/quantum && git log --oneline -10

echo ""
echo "=== AI ENGINE STARTUP LOGS (latest restart) ==="
journalctl -u quantum-ai-engine -n 500 --no-pager -q 2>/dev/null | grep -E '(Loaded.*pkl|Loaded.*pth|QSC.*ACTIVE|GOVERNMER|FAIL-CLOSED)' | tail -20

echo ""
echo "=== AI ENGINE ERRORS (last hour) ==="
journalctl -u quantum-ai-engine --since "1 hour ago" --no-pager -q 2>/dev/null | grep '"level": "ERROR"' | grep -v 'Dropping extras' | sed 's/.*"msg": "//' | sed 's/".*//' | sort | uniq -c | sort -rn | head -15

echo ""
echo "=== BACKEND SERVICE ENV ==="
grep -E 'Environment|AI_MAX|KELLY|LEVERAGE|DRAWDOWN|EQUITY|MAX_POS' /etc/systemd/system/quantum-backend.service 2>/dev/null | head -30

echo ""
echo "=== AI ENGINE SERVICE ENV ==="
grep -E 'Environment|AI_MAX|KELLY|LEVERAGE|DRAWDOWN|EQUITY|MAX_POS|FEATURES' /etc/systemd/system/quantum-ai-engine.service 2>/dev/null | head -30

echo ""
echo "=== REDIS RISK / EQUITY KEYS (all quantum: keys non-stream) ==="
redis-cli keys 'quantum:*' 2>/dev/null | grep -v 'stream\|group\|consumer\|event\|buffer\|ohlcv\|kline\|exchange\|position\|market\|trade\|paper\|order\|ticker' | sort

echo ""
echo "=== OPEN POSITION DETAILS ==="
for key in $(redis-cli keys 'quantum:position:[A-Z]*' 2>/dev/null | head -10); do
  echo "  $key: $(redis-cli get $key 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"side={d.get(\"side\",\"?\")} size={d.get(\"quantity\",d.get(\"size\",\"?\"))} entry={d.get(\"entry_price\",\"?\")}")' 2>/dev/null)"
done

echo ""
echo "=== DISK USAGE ==="
du -sh /opt/quantum/ai_engine/models/ 2>/dev/null
df -h / 2>/dev/null | tail -1

echo ""
echo "=== STALE OLD XGB MODELS (pre-v6) ==="
ls /opt/quantum/ai_engine/models/xgb_model_v*.pkl 2>/dev/null | wc -l
echo "xgb_model_v* count above (stale Nov 2025 snapshots - safe to delete)"

echo ""
echo "=== RL TRAINER LOGS ==="
journalctl -u quantum-rl-trainer -n 40 --no-pager -q 2>/dev/null | tail -10

echo ""
echo "=== BACKEND ERRORS (last hour) ==="
journalctl -u quantum-backend --since "1 hour ago" --no-pager -q 2>/dev/null | grep '"level": "ERROR"' | sed 's/.*"msg": "//' | sed 's/".*//' | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== RECENT TRADE DETAIL (last 3) ==="
redis-cli xrevrange quantum:stream:trade.closed + - COUNT 3 2>/dev/null

ENDSSH

#!/bin/bash
# Upload and run service.py patch script on VPS
set -e

SSH_CMD="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"
SCP_CMD="scp -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no"

echo "=== Uploading patch script to VPS ==="
$SCP_CMD /mnt/c/quantum_trader/patch_service.py root@46.224.116.254:/tmp/patch_service.py

echo ""
echo "=== Running patch on VPS ==="
$SSH_CMD "python3 /tmp/patch_service.py"

echo ""
echo "=== Verifying patches ==="
$SSH_CMD "
echo '--- TradeOutcome call ---'
grep -A 15 'outcome = TradeOutcome(' /opt/quantum/microservices/ai_engine/service.py | head -20

echo ''
echo '--- equity_usd fix (RL sizer) ---'
grep -A 4 'Read real account equity from Redis' /opt/quantum/microservices/ai_engine/service.py | head -12

echo ''
echo '--- Check no more 10000.0 hardcodes ---'
grep -n '10000\.0' /opt/quantum/microservices/ai_engine/service.py || echo 'No hardcoded 10000.0 found'
"

echo ""
echo "=== Committing Fix 2 + Fix 5 ==="
$SSH_CMD "
cd /opt/quantum
git add microservices/ai_engine/service.py
git commit -m 'fix(P1/P3): TradeOutcome 5 missing args + equity_usd from Redis'
git log --oneline -5
"

echo ""
echo "=== Restarting quantum-ai-engine ==="
$SSH_CMD "systemctl restart quantum-ai-engine && sleep 3 && systemctl is-active quantum-ai-engine"

echo ""
echo "=== Final verification (15s) ==="
sleep 15
$SSH_CMD "
echo '--- Service status ---'
systemctl is-active quantum-ai-engine
echo ''
echo '--- No RL learning failed errors (last 30s) ---'
journalctl -u quantum-ai-engine --since '-30s' --no-pager | grep -i 'rl signal\|learning failed\|TradeOutcome\|EMERGENCY_FLATTEN\|drawdown' | tail -20
echo ''
echo '--- Drawdown state ---'
redis-cli hgetall quantum:equity:current
echo ''
echo '--- AI_MAX_LEVERAGE in env ---'
grep AI_MAX_LEVERAGE /etc/quantum/ai-engine.env
"

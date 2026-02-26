#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Cleaning garbage Redis fields + final verify ==="
$SSH '
echo "--- Before cleanup ---"
redis-cli hgetall quantum:equity:current

echo ""
echo "--- Removing garbage fields ---"
# These fields were created by the failed PowerShell-interpolated commands
for field in echo New "peak:" "\\"; do
    redis-cli hdel quantum:equity:current "$field" 2>/dev/null && echo "Deleted: $field" || true
done

# Also check for any other non-expected fields
echo ""
echo "--- After cleanup ---"
redis-cli hgetall quantum:equity:current

echo ""
echo "=== Final system state ==="
echo "--- quantum-ai-engine status ---"
systemctl is-active quantum-ai-engine
echo ""
echo "--- Last 20 lines of ai-engine log ---"
journalctl -u quantum-ai-engine --since "-60s" --no-pager | tail -20
echo ""
echo "--- git log last 5 commits ---"
cd /opt/quantum && git log --oneline -5
echo ""
echo "--- AI_MAX_LEVERAGE ---"
grep AI_MAX_LEVERAGE /etc/quantum/ai-engine.env
echo ""
echo "--- equity:current drawdown ---"
EQ=$(redis-cli hget quantum:equity:current equity)
PK=$(redis-cli hget quantum:equity:current peak)
echo "equity=$EQ  peak=$PK  (both same = 0% drawdown)"
'

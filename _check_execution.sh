#!/bin/bash
# Check if intent_executor is actually processing harvest_v2 plans
echo "=== RECENT apply.result: filter for harvest_v2 results ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 30 | grep -A3 -B3 "harvest_v2" | head -60

echo ""
echo "=== INTENT_EXECUTOR service name ==="
systemctl list-units --type=service --state=active 2>/dev/null | grep -iE "intent|exec" | head -10

echo ""
echo "=== Check quantum-execution.service file ==="
cat /etc/systemd/system/quantum-execution.service | head -30

echo ""
echo "=== Tail intent-executor/execution logs ==="
journalctl -u quantum-execution.service -n 20 --no-pager 2>/dev/null || true
journalctl -u quantum-intent-executor.service -n 20 --no-pager 2>/dev/null || true

echo ""
echo "=== Recent apply.result: ALL last 10 ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10

echo ""
echo "=== Check harvest_v2 R thresholds in evaluator ==="
cat /opt/quantum/microservices/harvest_v2/engine/evaluator.py 2>/dev/null | head -80

echo ""
echo "=== harvest_v2 config loader ==="
cat /opt/quantum/microservices/harvest_v2/engine/config.py 2>/dev/null | head -60

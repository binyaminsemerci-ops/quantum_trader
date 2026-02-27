#!/bin/bash
echo "=== EXECUTION PIPELINE — DEEP INVESTIGATION ==="

echo ""
echo "--- 1. APPLY LAYER service file ---"
cat /etc/systemd/system/quantum-apply-layer.service | grep -E "Exec|Working|Env|User"

echo ""
echo "--- 2. INTENT EXECUTOR service file ---"
cat /etc/systemd/system/quantum-intent-executor.service | grep -E "Exec|Working|Env|User|EnvironmentFile"

echo ""
echo "--- 3. INTENT EXECUTOR env file ---"
cat /etc/quantum/intent-executor.env 2>/dev/null || cat /etc/quantum/intent_executor.env 2>/dev/null || echo "no env file found"

echo ""
echo "--- 4. EXECUTION service file ---"
cat /etc/systemd/system/quantum-execution.service | grep -E "Exec|Working|Env|User|EnvironmentFile"

echo ""
echo "--- 5. EXECUTION env file ---"
cat /etc/quantum/execution.env 2>/dev/null | grep -v "SECRET\|API_KEY" | head -30

echo ""
echo "--- 6. AUTONOMOUS TRADER logs (last 20) ---"
journalctl -u quantum-autonomous-trader -n 20 --no-pager

echo ""
echo "--- 7. INTENT EXECUTOR last 30 logs ---"
journalctl -u quantum-intent-executor -n 30 --no-pager

echo ""
echo "--- 8. HARVEST BRAIN last 20 logs ---"
journalctl -u quantum-harvest-brain -n 20 --no-pager

echo ""
echo "--- 9. WHO WRITES TO quantum:stream:trade.intent ---"
redis-cli xrevrange quantum:stream:trade.intent + - COUNT 3

echo ""
echo "--- 10. APPLY LAYER env file ---"
cat /etc/quantum/apply-layer.env 2>/dev/null | grep -v "SECRET\|API_KEY" | head -30

#!/usr/bin/env bash
# P7B shadow validation: wait for tick and capture audit entry for ETHUSDT

REDISCLI=$(which redis-cli 2>/dev/null || echo /usr/bin/redis-cli)

echo "=== P7B SHADOW VALIDATION: WAIT FOR TICK ==="
echo "redis-cli: $REDISCLI"
echo "Time: $(date --iso-8601=seconds)"
echo ""

# Capture baseline: last audit ID before tick
BEFORE=$($REDISCLI XREVRANGE quantum:stream:exit.audit + - COUNT 1 2>/dev/null | head -1)
echo "BASELINE_LAST_AUDIT_ID: $BEFORE"
echo ""

# Confirm position is in Redis
POS_EXISTS=$($REDISCLI EXISTS quantum:position:ETHUSDT 2>/dev/null)
echo "quantum:position:ETHUSDT EXISTS: $POS_EXISTS"

# Confirm ticker is in Redis
MARK=$($REDISCLI HGET quantum:ticker:ETHUSDT markPrice 2>/dev/null)
echo "quantum:ticker:ETHUSDT markPrice: $MARK"
echo ""

# Check service is running
IS_ACTIVE=$(systemctl is-active quantum-exit-management-agent 2>/dev/null)
echo "Service status: $IS_ACTIVE"
echo ""

echo "Waiting 15 seconds for service tick (loop_sec=5, need 3 cycles to be safe)..."
sleep 15

echo ""
echo "=== AFTER TICK: NEW AUDIT ENTRIES ==="

# Get all entries after baseline
if [ -z "$BEFORE" ]; then
    # No baseline, just get last 5
    ENTRIES=$($REDISCLI XREVRANGE quantum:stream:exit.audit + - COUNT 5 2>/dev/null)
else
    ENTRIES=$($REDISCLI XRANGE quantum:stream:exit.audit "$BEFORE" + 2>/dev/null)
fi
echo "$ENTRIES"
echo ""
echo "=== FILTER: ETHUSDT entries ==="
echo "$ENTRIES" | grep -A 50 "ETHUSDT" | head -80 || echo "No ETHUSDT entries found"

echo ""
echo "=== FILTER: P7B fields ==="
echo "$ENTRIES" | grep -E "patch|qwen3" || echo "No patch/qwen3 fields found"

echo ""
echo "=== SERVICE LOG LAST 30 LINES ==="
journalctl -u quantum-exit-management-agent --no-pager -n 20 2>/dev/null | grep -E "TICK|ETHUSDT|qwen3|scoring|PATCH|ERROR" || echo "No relevant log lines"

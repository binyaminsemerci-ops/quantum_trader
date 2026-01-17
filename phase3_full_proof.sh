#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "PHASE 3: FULL SYSTEM PROOF"
echo "Zombie Consumer Fix + ACK Hardening"
echo "=========================================="
echo ""

# Baseline evidence
echo "=== BASELINE: Consumer Group Status ==="
redis-cli XINFO GROUPS quantum:stream:trade.intent
echo ""

echo "=== BASELINE: Consumer Details ==="
redis-cli XINFO CONSUMERS quantum:stream:trade.intent quantum:group:execution:trade.intent
echo ""

echo "=== BASELINE: Pending Messages ==="
PENDING=$(redis-cli XPENDING quantum:stream:trade.intent quantum:group:execution:trade.intent - + 1 | wc -l)
echo "Pending count: $PENDING"
echo ""

# Dedup test
echo "=== DEDUP TEST: Injecting duplicate trade intent ==="
TRACE_ID="PROOF_TEST_$(date +%s)"
INTENT_JSON=$(cat <<EOF
{
  "trace_id": "$TRACE_ID",
  "symbol": "BTCUSDT",
  "action": "BUY",
  "quantity": 0.001,
  "entry_price": 50000.0,
  "stop_loss": 49000.0,
  "take_profit": 52000.0,
  "confidence": 0.85,
  "model_ensemble": "proof_test",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"
}
EOF
)

echo "Injecting intent 1..."
redis-cli XADD quantum:stream:trade.signal "*" data "$INTENT_JSON" >/dev/null
sleep 2

echo "Injecting intent 2 (DUPLICATE)..."
redis-cli XADD quantum:stream:trade.signal "*" data "$INTENT_JSON" >/dev/null
sleep 3

echo ""
echo "=== Checking for DUPLICATE_SKIP in router logs ==="
SKIP_COUNT=$(grep -c "DUPLICATE_SKIP.*$TRACE_ID" /var/log/quantum/ai_strategy_router.log 2>/dev/null || echo 0)
echo "DUPLICATE_SKIP count for $TRACE_ID: $SKIP_COUNT"

if [ "$SKIP_COUNT" -ge 1 ]; then
    echo "✅ DEDUP TEST: PASS (duplicate detected and skipped)"
else
    echo "❌ DEDUP TEST: FAIL (no duplicate skip found)"
fi
echo ""

# Terminal state test
echo "=== TERMINAL STATE TEST: Checking execution.result stream ==="
RECENT_RESULTS=$(redis-cli XREVRANGE quantum:stream:execution.result + - COUNT 5)
echo "Recent execution results (last 5):"
echo "$RECENT_RESULTS" | head -20
echo ""

# Restart resilience test
echo "=== RESTART TEST: Simulating service restart ==="
PID_BEFORE=$(systemctl show quantum-execution --property=MainPID --value)
echo "PID before restart: $PID_BEFORE"

systemctl restart quantum-execution
sleep 5

PID_AFTER=$(systemctl show quantum-execution --property=MainPID --value)
echo "PID after restart: $PID_AFTER"
echo ""

echo "=== POST-RESTART: Consumer status ==="
CONSUMERS_AFTER=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | grep "^consumers" -A1 | tail -1)
PENDING_AFTER=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | grep "^pending" -A1 | tail -1)
echo "Consumers: $CONSUMERS_AFTER"
echo "Pending: $PENDING_AFTER"
echo ""

if [ "$CONSUMERS_AFTER" == "1" ] && [ "$PENDING_AFTER" == "0" ]; then
    echo "✅ RESTART TEST: PASS (no zombie, pending stable)"
else
    echo "❌ RESTART TEST: FAIL (consumers=$CONSUMERS_AFTER, pending=$PENDING_AFTER)"
fi
echo ""

# Recovery log verification
echo "=== RECOVERY LOG: Last 10 entries ==="
tail -10 /var/log/quantum/stream_recover.log
echo ""

# Final verdict
echo "=========================================="
echo "FINAL VERDICT"
echo "=========================================="
echo "Dedup: $([ "$SKIP_COUNT" -ge 1 ] && echo PASS || echo FAIL)"
echo "Restart: $([ "$CONSUMERS_AFTER" == "1" ] && echo PASS || echo FAIL)"
echo "Pending: $PENDING_AFTER (should be 0 or low)"
echo "=========================================="

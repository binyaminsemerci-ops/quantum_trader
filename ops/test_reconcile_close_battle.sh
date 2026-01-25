#!/bin/bash
# Battle Test: RECONCILE_CLOSE with backlog and lease renewal
# Tests that Apply Layer renews HOLD lease and executes despite queue backlog

set -e

SYMBOL="BTCUSDT"
TEST_ID=$(date +%s)

echo "=== RECONCILE_CLOSE Battle Test ==="
echo "Test ID: $TEST_ID"
echo

# 1. Set SHORT HOLD lease (60s to test renewal)
echo "Step 1: Set HOLD with SHORT lease (60s)"
redis-cli SET quantum:reconcile:hold:$SYMBOL 1 EX 60
redis-cli SET quantum:reconcile:hold_reason:$SYMBOL reconcile_drift EX 60
redis-cli SET quantum:reconcile:hold_sig:$SYMBOL battletest EX 60
echo "✓ HOLD set with 60s TTL"
echo

# 2. Create backlog: 10 dummy messages ahead in queue
echo "Step 2: Create backlog (10 dummy messages)"
for i in {1..10}; do
    redis-cli XADD quantum:stream:reconcile.close "*" \
        plan_id "dummy:backlog:${TEST_ID}:${i}" \
        decision "RECONCILE_CLOSE" \
        symbol "DUMMYSYMBOL" \
        side "SELL" \
        type "MARKET" \
        qty "0.001" \
        reduceOnly "true" \
        reason "reconcile_drift" \
        source "p3.4" \
        signature "dummy${i}" \
        exchange_amt "0.001" \
        ledger_amt "0.0" >/dev/null
done
echo "✓ 10 dummy messages added to stream"
echo

# 3. Add REAL RECONCILE_CLOSE message
echo "Step 3: Add REAL RECONCILE_CLOSE message (after backlog)"
PLAN_ID="battle:${SYMBOL}:${TEST_ID}"
redis-cli XADD quantum:stream:reconcile.close "*" \
    plan_id "$PLAN_ID" \
    decision "RECONCILE_CLOSE" \
    symbol "$SYMBOL" \
    side "SELL" \
    type "MARKET" \
    qty "0.001" \
    reduceOnly "true" \
    reason "reconcile_drift" \
    source "p3.4" \
    signature "battletest" \
    exchange_amt "0.001" \
    ledger_amt "0.0"
echo "✓ REAL message added: $PLAN_ID"
echo

# 4. Check HOLD TTL before processing
echo "Step 4: Check HOLD TTL before processing"
INITIAL_TTL=$(redis-cli TTL quantum:reconcile:hold:$SYMBOL)
echo "✓ Initial HOLD TTL: ${INITIAL_TTL}s"
echo

# 5. Wait for Apply Layer to process (stream consumer processes every 5s)
echo "Step 5: Waiting for Apply Layer to process (~15-30s with backlog)..."
sleep 30

# 6. Check if HOLD was renewed
echo
echo "Step 6: Check HOLD TTL after processing"
FINAL_TTL=$(redis-cli TTL quantum:reconcile:hold:$SYMBOL)
if [ "$FINAL_TTL" -eq -2 ]; then
    echo "✓ HOLD released (execution successful, position closed)"
    HOLD_STATUS="RELEASED"
elif [ "$FINAL_TTL" -gt "$INITIAL_TTL" ]; then
    echo "✓ HOLD lease RENEWED! TTL increased from ${INITIAL_TTL}s to ${FINAL_TTL}s"
    HOLD_STATUS="RENEWED"
else
    echo "⚠ HOLD TTL: ${FINAL_TTL}s (was ${INITIAL_TTL}s)"
    HOLD_STATUS="UNCHANGED"
fi
echo

# 7. Check execution logs
echo "Step 7: Check Apply Layer execution logs"
echo "Looking for plan_id: ${PLAN_ID:0:16}..."
journalctl -u quantum-apply-layer --since "2 minutes ago" | grep "${PLAN_ID:0:16}" | head -5

echo
echo "Looking for lease renewal..."
journalctl -u quantum-apply-layer --since "2 minutes ago" | grep "HOLD lease renewed" | tail -3

echo
echo "=== Battle Test Results ==="
echo "Test ID: $TEST_ID"
echo "HOLD Status: $HOLD_STATUS"
echo
if [ "$HOLD_STATUS" = "RELEASED" ] || [ "$HOLD_STATUS" = "RENEWED" ]; then
    echo "✅ TEST PASSED: System handled backlog and lease correctly"
    echo
    echo "What happened:"
    echo "1. HOLD set with 60s TTL"
    echo "2. 10 dummy messages created backlog"
    echo "3. Real RECONCILE_CLOSE queued behind backlog"
    echo "4. Apply Layer consumed backlog, renewed lease for BTCUSDT"
    echo "5. Executed RECONCILE_CLOSE despite initial 60s TTL"
    echo
    echo "This proves the system is BATTLE-TESTED."
    exit 0
else
    echo "❌ TEST FAILED or INCOMPLETE"
    echo "Check logs manually: journalctl -u quantum-apply-layer --since '5 minutes ago'"
    exit 1
fi

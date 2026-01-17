#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "P0 FIX PACK — PHASE 1: IDEMPOTENCY PROOF TEST"
echo "========================================================================"
echo ""

echo "[TEST] Injecting duplicate ai.decision events with same trace_id..."
echo ""

NONCE=$(date +%s%N | md5sum | cut -c1-8)
TRACE_ID="proof-test-$NONCE"
TS=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

echo "Trace ID: $TRACE_ID"
echo "Symbol: IDEMPTEST"
echo ""

# Inject first event
echo "[1/2] Injecting first event..."
EVENT1=$(redis-cli XADD quantum:stream:ai.decision.made "*" \
    event_type "ai.decision.made" \
    payload "{\"symbol\":\"IDEMPTEST\",\"side\":\"buy\",\"confidence\":0.96,\"entry_price\":100.0,\"quantity\":1.0,\"leverage\":1,\"stop_loss\":95.0,\"take_profit\":105.0,\"timestamp\":\"$TS\",\"test\":\"idempotency_proof\",\"model\":\"proof\"}" \
    trace_id "$TRACE_ID" \
    correlation_id "proof-corr-$NONCE" \
    timestamp "$TS" \
    source "idempotency-proof-test")
echo "✓ Event 1 ID: $EVENT1"
sleep 2

# Inject duplicate event (same trace_id)
echo ""
echo "[2/2] Injecting duplicate event (SAME trace_id)..."
EVENT2=$(redis-cli XADD quantum:stream:ai.decision.made "*" \
    event_type "ai.decision.made" \
    payload "{\"symbol\":\"IDEMPTEST\",\"side\":\"buy\",\"confidence\":0.96,\"entry_price\":100.0,\"quantity\":1.0,\"leverage\":1,\"stop_loss\":95.0,\"take_profit\":105.0,\"timestamp\":\"$TS\",\"test\":\"idempotency_proof_DUP\",\"model\":\"proof\"}" \
    trace_id "$TRACE_ID" \
    correlation_id "proof-corr-$NONCE" \
    timestamp "$TS" \
    source "idempotency-proof-test")
echo "✓ Event 2 ID: $EVENT2"

echo ""
echo "Waiting 10s for router processing..."
sleep 10

echo ""
echo "========================================================================"
echo "PROOF VERIFICATION"
echo "========================================================================"
echo ""

# Check dedup key
echo "[1/4] Checking Redis dedup key..."
DEDUP_KEY="quantum:dedup:trade_intent:$TRACE_ID"
if redis-cli EXISTS "$DEDUP_KEY" | grep -q 1; then
    echo "✓ Dedup key exists: $DEDUP_KEY"
    TTL=$(redis-cli TTL "$DEDUP_KEY")
    echo "  TTL: ${TTL}s (should be ~86400)"
else
    echo "❌ Dedup key NOT found!"
fi

echo ""
echo "[2/4] Checking router logs for DUPLICATE_SKIP..."
ROUTER_LOGS=$(journalctl -u quantum-ai-strategy-router.service -n 100 --no-pager | grep -i "$TRACE_ID" || echo "No logs found")
echo "$ROUTER_LOGS"

if echo "$ROUTER_LOGS" | grep -q "DUPLICATE_SKIP"; then
    echo ""
    echo "✅ Router correctly SKIPPED duplicate!"
else
    echo ""
    echo "⚠️  No DUPLICATE_SKIP found in logs"
fi

echo ""
echo "[3/4] Counting trade.intent events for IDEMPTEST..."
INTENT_COUNT=$(redis-cli XRANGE quantum:stream:trade.intent - + | grep -c "IDEMPTEST" || echo "0")
echo "Trade intents created: $INTENT_COUNT"

if [ "$INTENT_COUNT" -eq 1 ]; then
    echo "✅ PASS - Only 1 trade intent created (idempotency works!)"
elif [ "$INTENT_COUNT" -eq 0 ]; then
    echo "⚠️  WARNING - 0 intents (may be filtered by Strategy Brain)"
else
    echo "❌ FAIL - $INTENT_COUNT intents created (should be 1)"
fi

echo ""
echo "[4/4] Sample trade.intent event..."
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 | grep -A 5 "IDEMPTEST" | head -10 || echo "(No IDEMPTEST intent found)"

echo ""
echo "========================================================================"
echo "IDEMPOTENCY PROOF TEST COMPLETE"
echo "========================================================================"
echo ""

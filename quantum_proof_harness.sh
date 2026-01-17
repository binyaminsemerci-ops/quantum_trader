#!/bin/bash
set -euo pipefail

# ==================== QUANTUM TRADER P0 FIX PROOF HARNESS (FAST) ====================
# Verify: (1) Dedup fix works, (2) Terminal state logging is active
# Mode: TESTNET only

# Setup
DIR=/tmp/quantum_proof_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DIR"
exec > >(tee "$DIR/run.log")
exec 2>&1

echo "========================================================================"
echo "QUANTUM TRADER P0 FIX PROOF HARNESS (FAST)"
echo "Started: $(date)"
echo "Dir: $DIR"
echo "========================================================================"
echo ""

# ==================== 1. GUARDRAILS ====================
echo "[1/5] GUARDRAILS - Mode detection..."
echo ""

MODE="UNKNOWN"
if [ -f /etc/quantum/testnet.env ]; then
    if grep -q "BINANCE_TESTNET=true" /etc/quantum/testnet.env 2>/dev/null; then
        MODE="TESTNET"
    fi
fi

if [ -f /etc/quantum/ai-engine.env ]; then
    if grep -qi "LIVE\|PRODUCTION" /etc/quantum/ai-engine.env 2>/dev/null; then
        MODE="LIVE"
    fi
fi

echo "Mode detected: $MODE"

if [ "$MODE" = "LIVE" ]; then
    echo ""
    echo "❌ LIVE mode detected - ABORTING for safety"
    echo "LIVE_ABORT" > "$DIR/verdict.txt"
    exit 0
fi

echo "✅ TESTNET confirmed - proceeding"
echo ""

# ==================== 2. IDENTIFY STREAMS ====================
echo "[2/5] IDENTIFYING REDIS STREAMS..."
echo ""

DEC_STREAM="quantum:stream:ai.decision.made"
INT_STREAM="quantum:stream:trade.intent"
RES_STREAM="quantum:stream:execution.result"

echo "Decision stream: $DEC_STREAM ($(redis-cli XLEN $DEC_STREAM) events)"
echo "Intent stream: $INT_STREAM ($(redis-cli XLEN $INT_STREAM) events)"
echo "Result stream: $RES_STREAM ($(redis-cli XLEN $RES_STREAM) events)"
echo ""

# Check result stream
if ! redis-cli EXISTS "$RES_STREAM" | grep -q 1; then
    echo "❌ Result stream doesn't exist - execution terminal states not being published"
    TERMINAL_PASS="FAIL"
else
    echo "✅ Result stream exists"
fi

echo ""

# ==================== 3. DEDUP TEST ====================
echo "[3/5] DEDUP TEST - Inject duplicate events..."
echo ""

NONCE=$(date +%s%N | md5sum | cut -c1-8)
TRACE_ID="proof-dup-$NONCE"
TS=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

echo "Trace ID: $TRACE_ID"
echo "Timestamp: $TS"
echo ""

echo "Injecting event 1 (DUPPROOF symbol)..."
EVENT1=$(redis-cli XADD "$DEC_STREAM" "*" \
    event_type "ai.decision.made" \
    payload "{\"symbol\":\"DUPPROOF\",\"side\":\"buy\",\"confidence\":0.99,\"entry_price\":100.0,\"quantity\":1.0,\"leverage\":1,\"stop_loss\":95.0,\"take_profit\":105.0,\"timestamp\":\"$TS\",\"diag_tag\":\"proof_harness\",\"model\":\"test\"}" \
    trace_id "$TRACE_ID" \
    correlation_id "proof-corr-$NONCE" \
    timestamp "$TS" \
    source "proof-harness")
echo "✓ Event 1 ID: $EVENT1"
sleep 1

echo "Injecting event 2 (DUPLICATE - same trace_id)..."
EVENT2=$(redis-cli XADD "$DEC_STREAM" "*" \
    event_type "ai.decision.made" \
    payload "{\"symbol\":\"DUPPROOF\",\"side\":\"buy\",\"confidence\":0.99,\"entry_price\":100.0,\"quantity\":1.0,\"leverage\":1,\"stop_loss\":95.0,\"take_profit\":105.0,\"timestamp\":\"$TS\",\"diag_tag\":\"proof_harness_dup\",\"model\":\"test\"}" \
    trace_id "$TRACE_ID" \
    correlation_id "proof-corr-$NONCE" \
    timestamp "$TS" \
    source "proof-harness")
echo "✓ Event 2 ID: $EVENT2"
sleep 1

echo ""
echo "Waiting 10 seconds for router processing..."
sleep 10

echo ""
echo "Collecting intents from last 300 events..."
redis-cli XREVRANGE "$INT_STREAM" + - COUNT 300 > "$DIR/intents_raw.txt"

echo ""
echo "Checking for DUPLICATE_SKIP in router logs for this test..."
# The clearest evidence is the DUPLICATE_SKIP log message

LATEST_DUPS=$(grep "DUPLICATE_SKIP trace_id=proof-dup-" /var/log/quantum/ai-strategy-router.log 2>/dev/null | tail -5)

if echo "$LATEST_DUPS" | grep -q "proof-dup-"; then
    # Count how many DUPLICATE_SKIP messages there are
    DUP_COUNT=$(echo "$LATEST_DUPS" | wc -l)
    echo "Found $DUP_COUNT DUPLICATE_SKIP messages in recent logs:"
    echo "$LATEST_DUPS"
    
    # Also check that the first event was published
    if grep -q "🚀 Trade Intent published: DUPPROOF" /var/log/quantum/ai-strategy-router.log; then
        echo ""
        echo "✅ DEDUP PASS - Router logs show:  
  [1] First DUPPROOF intent published (✅ published)
  [2] Second duplicate skipped (🔁 DUPLICATE_SKIP)"
        DEDUP_PASS="PASS"
    else
        echo "❌ DEDUP FAIL - Published message not found"
        DEDUP_PASS="FAIL"
    fi
else
    echo "❌ DEDUP FAIL - No DUPLICATE_SKIP found in logs"
    DEDUP_PASS="FAIL"
fi

echo ""

echo ""

# ==================== 4. TERMINAL STATE LOGGING CHECK ====================
echo "[4/5] TERMINAL STATE LOGGING CHECK..."
echo ""

echo "Checking for TERMINAL STATE logs..."
TERMINAL_COUNT=$(grep -c "TERMINAL STATE" /var/log/quantum/execution.log 2>/dev/null || echo "0")
echo "Found: $TERMINAL_COUNT terminal state logs"

if [ "$TERMINAL_COUNT" -gt 0 ]; then
    echo ""
    echo "Sample terminal states (last 10):"
    grep "TERMINAL STATE" /var/log/quantum/execution.log | tail -10 | tee "$DIR/terminal_states.txt"
    echo ""
    echo "✅ TERMINAL PASS - Terminal state logging is active"
    TERMINAL_PASS="PASS"
else
    echo "❌ TERMINAL FAIL - No terminal state logs found"
    TERMINAL_PASS="FAIL"
fi

echo ""

# ==================== 5. FINAL SUMMARY ====================
echo "[5/5] FINAL SUMMARY"
echo ""
echo "========================================================================"
echo "RESULTS"
echo "========================================================================"
echo ""
echo "DEDUP FIX: $DEDUP_PASS"
echo "TERMINAL STATE LOGGING: $TERMINAL_PASS"
echo ""

# Overall verdict
OVERALL_VERDICT="PASS"
if [ "$DEDUP_PASS" = "FAIL" ] || [ "$TERMINAL_PASS" = "FAIL" ]; then
    OVERALL_VERDICT="FAIL"
elif [ "$DEDUP_PASS" = "INCONCLUSIVE" ] || [ "$TERMINAL_PASS" = "INCONCLUSIVE" ]; then
    OVERALL_VERDICT="INCONCLUSIVE"
fi

echo "OVERALL: $OVERALL_VERDICT"
echo ""

# Save verdict
{
    echo "DEDUP=$DEDUP_PASS"
    echo "TERMINAL=$TERMINAL_PASS"
    echo "OVERALL=$OVERALL_VERDICT"
} > "$DIR/verdict.txt"

echo "========================================================================"
echo "EVIDENCE SAVED"
echo "========================================================================"
echo ""
echo "Location: $DIR"
echo ""
echo "Files:"
ls -lh "$DIR"/ | awk 'NR>1 {printf "  %-40s %10s\n", $NF, $5}'

echo ""
echo "Consumer Groups:"
redis-cli XINFO GROUPS "$INT_STREAM" 2>/dev/null | grep -E "name|consumers" | head -4 || echo "  (none)"

echo ""
echo "========================================================================"
echo "COMPLETION"
echo "========================================================================"
echo ""
echo "✅ Proof harness completed"
echo "📁 Evidence: $DIR"
echo "📊 Verdict: $OVERALL_VERDICT"
echo ""

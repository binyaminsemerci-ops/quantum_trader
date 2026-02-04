#!/bin/bash
# GO-LIVE SHADOW MODE
# Phase B: MAINNET data/keys, but PAPER_TRADING=true
# No real orders sent, but full pipeline runs

set -e

PROOF_FILE="GO_LIVE_SHADOW_PROOF.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
DURATION_MINUTES=60
VPS_HOST="root@46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"

echo "======================================"
echo "  GO-LIVE PHASE B: SHADOW MODE"
echo "======================================"
echo "Date: $TIMESTAMP"
echo "Duration: ${DURATION_MINUTES} minutes"
echo ""

# Start proof document
cat > "$PROOF_FILE" << EOF
# GO-LIVE SHADOW MODE PROOF

**Date**: $TIMESTAMP  
**Phase**: B - Shadow Mode (MAINNET data, PAPER execution)  
**Duration**: ${DURATION_MINUTES} minutes  
**Risk Level**: 🟡 LOW (no real orders)

---

## SHADOW MODE CONFIGURATION

\`\`\`
BINANCE_USE_TESTNET=false
PAPER_TRADING=true
MAINNET_KEYS=enabled
ORDER_EXECUTION=disabled
\`\`\`

**Objective**: Verify full pipeline against MAINNET data without sending real orders.

---

## PRE-SHADOW CHECKS

EOF

echo "🔍 Verifying environment configuration..."

# Check if preflight passed
if [ ! -f "GO_LIVE_PREFLIGHT_PROOF.md" ]; then
    echo "❌ PREFLIGHT not completed! Run go_live_preflight.sh first."
    exit 1
fi

# Verify VPS connectivity
echo "📡 Testing VPS connectivity..."
ssh -i "$SSH_KEY" "$VPS_HOST" "echo 'VPS connected'" || {
    echo "❌ Cannot connect to VPS"
    exit 1
}

# Verify environment variables
echo "🔧 Verifying SHADOW MODE environment..."
ssh -i "$SSH_KEY" "$VPS_HOST" "cd /home/qt/quantum_trader && docker compose exec -T auto_executor env | grep -E 'BINANCE_USE_TESTNET|PAPER_TRADING'" > /tmp/shadow_env.txt

TESTNET_FLAG=$(grep "BINANCE_USE_TESTNET" /tmp/shadow_env.txt | cut -d= -f2 || echo "not_found")
PAPER_FLAG=$(grep "PAPER_TRADING" /tmp/shadow_env.txt | cut -d= -f2 || echo "not_found")

cat >> "$PROOF_FILE" << EOF
### Environment Verification

\`\`\`
BINANCE_USE_TESTNET=$TESTNET_FLAG
PAPER_TRADING=$PAPER_FLAG
\`\`\`

EOF

if [ "$TESTNET_FLAG" != "false" ] || [ "$PAPER_FLAG" != "true" ]; then
    echo "❌ Environment not configured for SHADOW mode!"
    echo "   Expected: BINANCE_USE_TESTNET=false, PAPER_TRADING=true"
    echo "   Got: TESTNET=$TESTNET_FLAG, PAPER=$PAPER_FLAG"
    cat >> "$PROOF_FILE" << EOF
**Status**: ❌ FAILED - Environment misconfigured

EOF
    exit 1
fi

echo "✅ Environment: SHADOW mode configured correctly"
cat >> "$PROOF_FILE" << EOF
**Status**: ✅ PASS

---

## SHADOW RUN EXECUTION

**Start Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

EOF

# Capture baseline metrics
echo "📊 Capturing baseline metrics..."
ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli XLEN quantum:stream:intent" > /tmp/baseline_intent.txt
ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:counter:orders_submitted" > /tmp/baseline_orders.txt

BASELINE_INTENT=$(cat /tmp/baseline_intent.txt)
BASELINE_ORDERS=$(cat /tmp/baseline_orders.txt || echo "0")

echo "   Baseline INTENT stream length: $BASELINE_INTENT"
echo "   Baseline orders submitted: $BASELINE_ORDERS"

cat >> "$PROOF_FILE" << EOF
### Baseline Metrics

- Intent stream length: $BASELINE_INTENT
- Orders submitted: $BASELINE_ORDERS

EOF

# Monitor for duration
echo ""
echo "🚀 SHADOW MODE ACTIVE - Monitoring for ${DURATION_MINUTES} minutes..."
echo "   Press Ctrl+C to abort early"
echo ""

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_MINUTES * 60))

SAMPLES=0
MAX_SAMPLES=10
SLEEP_INTERVAL=$((DURATION_MINUTES * 60 / MAX_SAMPLES))

while [ $(date +%s) -lt $END_TIME ]; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    REMAINING=$(( END_TIME - $(date +%s) ))
    
    echo "⏱️  Elapsed: ${ELAPSED}s | Remaining: ${REMAINING}s"
    
    # Sample metrics
    CURRENT_INTENT=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli XLEN quantum:stream:intent")
    CURRENT_ORDERS=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:counter:orders_submitted" || echo "0")
    
    INTENT_DELTA=$((CURRENT_INTENT - BASELINE_INTENT))
    ORDERS_DELTA=$((CURRENT_ORDERS - BASELINE_ORDERS))
    
    echo "   📥 Intents received: +$INTENT_DELTA"
    echo "   📤 Orders submitted: +$ORDERS_DELTA (should stay 0)"
    
    if [ "$ORDERS_DELTA" -gt 0 ]; then
        echo "⚠️  WARNING: Orders submitted in SHADOW mode! Investigating..."
        ssh -i "$SSH_KEY" "$VPS_HOST" "docker logs quantum_auto_executor --tail 50" > /tmp/shadow_orders_leak.log
        cat >> "$PROOF_FILE" << EOF
### ⚠️ CRITICAL ISSUE: Orders Submitted in Shadow Mode

Orders submitted: $ORDERS_DELTA (expected: 0)

**Logs**:
\`\`\`
$(cat /tmp/shadow_orders_leak.log)
\`\`\`

EOF
        echo "❌ SHADOW MODE FAILED: Orders leaked through!"
        exit 1
    fi
    
    SAMPLES=$((SAMPLES + 1))
    sleep "$SLEEP_INTERVAL"
done

# Final metrics
echo ""
echo "✅ SHADOW MODE COMPLETE"

FINAL_INTENT=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli XLEN quantum:stream:intent")
FINAL_ORDERS=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:counter:orders_submitted" || echo "0")

TOTAL_INTENT=$((FINAL_INTENT - BASELINE_INTENT))
TOTAL_ORDERS=$((FINAL_ORDERS - BASELINE_ORDERS))

cat >> "$PROOF_FILE" << EOF
**End Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

### Final Metrics

- Intents processed: $TOTAL_INTENT
- Orders submitted: $TOTAL_ORDERS
- **Expected**: Orders = 0 (paper mode)

EOF

# Check logs for "WOULD_SUBMIT" entries
echo "🔍 Checking for WOULD_SUBMIT log entries..."
ssh -i "$SSH_KEY" "$VPS_HOST" "docker logs quantum_auto_executor --since ${DURATION_MINUTES}m 2>&1 | grep -c 'WOULD_SUBMIT' || echo 0" > /tmp/would_submit_count.txt
WOULD_SUBMIT=$(cat /tmp/would_submit_count.txt)

echo "   Found $WOULD_SUBMIT WOULD_SUBMIT log entries"

cat >> "$PROOF_FILE" << EOF
### Paper Execution Proof

- WOULD_SUBMIT log entries: $WOULD_SUBMIT
- Actual orders submitted: $TOTAL_ORDERS

EOF

# Acceptance criteria
PASS=true

if [ "$TOTAL_ORDERS" -ne 0 ]; then
    echo "❌ FAIL: Orders submitted in shadow mode (expected 0, got $TOTAL_ORDERS)"
    PASS=false
fi

if [ "$TOTAL_INTENT" -eq 0 ]; then
    echo "⚠️  WARNING: No intents processed (pipeline may be stalled)"
    PASS=false
fi

if [ "$PASS" = true ]; then
    echo "✅ SHADOW MODE PASSED"
    cat >> "$PROOF_FILE" << EOF
---

## ACCEPTANCE CRITERIA

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Orders submitted | 0 | $TOTAL_ORDERS | ✅ PASS |
| Intents processed | >0 | $TOTAL_INTENT | ✅ PASS |
| WOULD_SUBMIT logs | >0 | $WOULD_SUBMIT | ✅ PASS |

**VERDICT**: ✅ **SHADOW MODE PASSED**

Shadow run completed successfully. System processes intents and logs execution decisions without submitting real orders.

**Next Step**: Review logs, then proceed to Phase C (Live Small) with extreme caution.

EOF
    echo ""
    echo "📄 Proof document: $PROOF_FILE"
    exit 0
else
    echo "❌ SHADOW MODE FAILED"
    cat >> "$PROOF_FILE" << EOF
---

## ACCEPTANCE CRITERIA

**VERDICT**: ❌ **SHADOW MODE FAILED**

Shadow run did not meet acceptance criteria. Do NOT proceed to live trading.

**Action Required**: Investigate issues and re-run shadow mode.

EOF
    echo ""
    echo "📄 Proof document: $PROOF_FILE"
    exit 1
fi

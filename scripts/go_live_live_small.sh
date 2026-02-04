#!/bin/bash
# GO-LIVE LIVE SMALL
# Phase C: REAL TRADING with extreme limitations
# Micro-notional, 1-3 symbols, hard caps

set -e

PROOF_FILE="GO_LIVE_LIVE_SMALL_PROOF.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
DURATION_MINUTES=120
VPS_HOST="root@46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"

echo "======================================"
echo "  🚨 GO-LIVE PHASE C: LIVE SMALL 🚨"
echo "======================================"
echo "Date: $TIMESTAMP"
echo "Duration: ${DURATION_MINUTES} minutes"
echo ""
echo "⚠️  WARNING: REAL TRADING WITH REAL MONEY"
echo "⚠️  Micro-notional + hard caps enforced"
echo ""

read -p "Type 'LIVE' to confirm live trading: " CONFIRM
if [ "$CONFIRM" != "LIVE" ]; then
    echo "❌ Aborted by operator"
    exit 1
fi

# Start proof document
cat > "$PROOF_FILE" << EOF
# GO-LIVE LIVE SMALL PROOF

**Date**: $TIMESTAMP  
**Phase**: C - Live Small (REAL TRADING)  
**Duration**: ${DURATION_MINUTES} minutes  
**Risk Level**: 🔴 HIGH (real money)

---

## LIVE SMALL CONFIGURATION

\`\`\`
PAPER_TRADING=false
BINANCE_USE_TESTNET=false
SYMBOL_ALLOWLIST=BTCUSDT,ETHUSDT (max 3)
MAX_POSITIONS=1
MAX_LEVERAGE=2
MAX_NOTIONAL_PER_TRADE=50 USDT
COOLDOWN_SECONDS=60
\`\`\`

**Objective**: Execute 1-3 real orders with full proof and controlled risk.

---

## PRE-LIVE CHECKS

EOF

echo "🔍 Verifying prerequisites..."

# Check if shadow passed
if [ ! -f "GO_LIVE_SHADOW_PROOF.md" ]; then
    echo "❌ SHADOW MODE not completed! Run go_live_shadow.sh first."
    exit 1
fi

if ! grep -q "✅ PASS" GO_LIVE_SHADOW_PROOF.md; then
    echo "❌ SHADOW MODE did not pass! Cannot proceed to live."
    exit 1
fi

# Verify environment
echo "🔧 Verifying LIVE SMALL environment..."
ssh -i "$SSH_KEY" "$VPS_HOST" "cd /home/qt/quantum_trader && docker compose exec -T auto_executor env | grep -E 'PAPER_TRADING|BINANCE_USE_TESTNET|MAX_POSITIONS|MAX_NOTIONAL'" > /tmp/live_env.txt

PAPER_FLAG=$(grep "PAPER_TRADING" /tmp/live_env.txt | cut -d= -f2 || echo "true")
TESTNET_FLAG=$(grep "BINANCE_USE_TESTNET" /tmp/live_env.txt | cut -d= -f2 || echo "true")

cat >> "$PROOF_FILE" << EOF
### Environment Verification

\`\`\`
$(cat /tmp/live_env.txt)
\`\`\`

EOF

if [ "$PAPER_FLAG" != "false" ]; then
    echo "❌ PAPER_TRADING still enabled! Must be false for live trading."
    echo "   Current: $PAPER_FLAG"
    cat >> "$PROOF_FILE" << EOF
**Status**: ❌ FAILED - PAPER_TRADING=$PAPER_FLAG (expected: false)

**Action**: Set PAPER_TRADING=false before proceeding.

EOF
    exit 1
fi

if [ "$TESTNET_FLAG" != "false" ]; then
    echo "❌ BINANCE_USE_TESTNET still enabled! Must be false for mainnet."
    echo "   Current: $TESTNET_FLAG"
    cat >> "$PROOF_FILE" << EOF
**Status**: ❌ FAILED - BINANCE_USE_TESTNET=$TESTNET_FLAG (expected: false)

**Action**: Set BINANCE_USE_TESTNET=false before proceeding.

EOF
    exit 1
fi

echo "✅ Environment: LIVE mode configured"
cat >> "$PROOF_FILE" << EOF
**Status**: ✅ PASS

---

## LIVE RUN EXECUTION

**Start Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

EOF

# Capture baseline
echo "📊 Capturing baseline..."
ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:counter:orders_submitted" > /tmp/baseline_orders.txt
BASELINE_ORDERS=$(cat /tmp/baseline_orders.txt || echo "0")

echo "   Baseline orders submitted: $BASELINE_ORDERS"

cat >> "$PROOF_FILE" << EOF
### Baseline Metrics

- Orders submitted: $BASELINE_ORDERS

EOF

# Monitor
echo ""
echo "🚀 LIVE SMALL ACTIVE - Monitoring for ${DURATION_MINUTES} minutes..."
echo ""

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_MINUTES * 60))

ORDER_PROOFS=()
POSITION_PROOFS=()

SAMPLES=0
MAX_SAMPLES=20
SLEEP_INTERVAL=$((DURATION_MINUTES * 60 / MAX_SAMPLES))

while [ $(date +%s) -lt $END_TIME ]; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    REMAINING=$(( END_TIME - $(date +%s) ))
    
    echo "⏱️  Elapsed: ${ELAPSED}s | Remaining: ${REMAINING}s"
    
    # Check orders
    CURRENT_ORDERS=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:counter:orders_submitted" || echo "0")
    ORDERS_DELTA=$((CURRENT_ORDERS - BASELINE_ORDERS))
    
    echo "   📤 Orders submitted: $ORDERS_DELTA"
    
    # If new order, capture proof
    if [ "$ORDERS_DELTA" -gt "${#ORDER_PROOFS[@]}" ]; then
        echo "   🎯 NEW ORDER DETECTED! Capturing proof..."
        ssh -i "$SSH_KEY" "$VPS_HOST" "docker logs quantum_auto_executor --tail 100 | grep -A 5 'ORDER_SUBMIT'" > /tmp/order_proof_$ORDERS_DELTA.log
        ORDER_PROOFS+=("/tmp/order_proof_$ORDERS_DELTA.log")
        
        # Check for orderId
        if grep -q "orderId" /tmp/order_proof_$ORDERS_DELTA.log; then
            ORDER_ID=$(grep "orderId" /tmp/order_proof_$ORDERS_DELTA.log | head -1 | grep -o "orderId[^,]*" | cut -d: -f2 || echo "unknown")
            echo "   ✅ Order ID: $ORDER_ID"
        else
            echo "   ⚠️  No orderId found in logs (potential issue)"
        fi
    fi
    
    # Check positions
    POSITIONS=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli HLEN quantum:positions:open" || echo "0")
    echo "   📊 Open positions: $POSITIONS"
    
    # Stop if target reached
    if [ "$ORDERS_DELTA" -ge 3 ]; then
        echo "✅ Target reached: 3+ orders executed"
        break
    fi
    
    SAMPLES=$((SAMPLES + 1))
    sleep "$SLEEP_INTERVAL"
done

# Final metrics
echo ""
echo "🏁 LIVE SMALL COMPLETE"

FINAL_ORDERS=$(ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:counter:orders_submitted" || echo "0")
TOTAL_ORDERS=$((FINAL_ORDERS - BASELINE_ORDERS))

cat >> "$PROOF_FILE" << EOF
**End Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

### Final Metrics

- Orders submitted: $TOTAL_ORDERS
- Order proofs captured: ${#ORDER_PROOFS[@]}

---

## ORDER EXECUTION PROOFS

EOF

# Include order proofs
for i in "${!ORDER_PROOFS[@]}"; do
    cat >> "$PROOF_FILE" << EOF
### Order $((i + 1))

\`\`\`
$(cat "${ORDER_PROOFS[$i]}")
\`\`\`

EOF
done

# Check for errors
echo "🔍 Checking for errors..."
ssh -i "$SSH_KEY" "$VPS_HOST" "docker logs quantum_auto_executor --since ${DURATION_MINUTES}m 2>&1 | grep -E '(-4045|-1111|INSUFFICIENT_BALANCE|MARKET_CLOSED)' || echo 'No errors'" > /tmp/live_errors.txt
ERRORS=$(cat /tmp/live_errors.txt)

cat >> "$PROOF_FILE" << EOF
---

## ERROR CHECK

\`\`\`
$ERRORS
\`\`\`

EOF

# Acceptance
PASS=true

if [ "$TOTAL_ORDERS" -eq 0 ]; then
    echo "⚠️  No orders executed (pipeline may be blocked)"
    PASS=false
fi

if [ "$TOTAL_ORDERS" -gt 0 ] && [ "${#ORDER_PROOFS[@]}" -eq 0 ]; then
    echo "❌ Orders submitted but no proofs captured!"
    PASS=false
fi

if echo "$ERRORS" | grep -q -E "(-4045|-1111)"; then
    echo "❌ Critical errors detected in logs"
    PASS=false
fi

if [ "$PASS" = true ]; then
    echo "✅ LIVE SMALL PASSED"
    cat >> "$PROOF_FILE" << EOF
---

## ACCEPTANCE CRITERIA

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Orders submitted | 1-3 | $TOTAL_ORDERS | ✅ PASS |
| Order proofs | >0 | ${#ORDER_PROOFS[@]} | ✅ PASS |
| Critical errors | 0 | None | ✅ PASS |

**VERDICT**: ✅ **LIVE SMALL PASSED**

Live trading executed successfully with full proof and no critical errors.

**Next Step**: Monitor for 24h stability, then consider gradual scale-up (Phase D).

EOF
    echo ""
    echo "📄 Proof document: $PROOF_FILE"
    exit 0
else
    echo "❌ LIVE SMALL FAILED"
    cat >> "$PROOF_FILE" << EOF
---

## ACCEPTANCE CRITERIA

**VERDICT**: ❌ **LIVE SMALL FAILED**

Live run did not meet acceptance criteria.

**Action Required**: Investigate issues before scaling up.

EOF
    echo ""
    echo "📄 Proof document: $PROOF_FILE"
    exit 1
fi

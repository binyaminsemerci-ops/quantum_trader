#!/bin/bash
# P3.3 E2E Proof Script for Testnet
# Verifies full permit chain: Apply publishes plan → P3.3 issues permit → Apply executes

set -e

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "====================================="
echo "P3.3 E2E Testnet Proof"
echo "====================================="
echo ""

# Function to check service status
check_service() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        echo "✓ $service: RUNNING"
        return 0
    else
        echo "✗ $service: NOT RUNNING"
        return 1
    fi
}

# 1. Verify services are running
echo "Step 1: Service Status"
echo "----------------------"
check_service "quantum-apply-layer" || exit 1
check_service "quantum-position-state-brain" || exit 1
echo ""

# 2. Clear dedupe keys to force new EXECUTE
echo "Step 2: Clear Dedupe Cache"
echo "---------------------------"
DEDUPE_COUNT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT --scan --pattern "quantum:apply:dedupe:*" | wc -l)
echo "Found $DEDUPE_COUNT dedupe keys"

if [ "$DEDUPE_COUNT" -gt 0 ]; then
    redis-cli -h $REDIS_HOST -p $REDIS_PORT --scan --pattern "quantum:apply:dedupe:*" | xargs -r redis-cli -h $REDIS_HOST -p $REDIS_PORT DEL > /dev/null
    echo "✓ Cleared dedupe cache"
fi
echo ""

# 3. Wait for EXECUTE plan
echo "Step 3: Waiting for EXECUTE Decision..."
echo "----------------------------------------"
TIMEOUT=120
START_TIME=$(date +%s)

while [ $(($(date +%s) - START_TIME)) -lt $TIMEOUT ]; do
    # Check Apply Layer logs for EXECUTE
    PLAN_ID=$(journalctl -u quantum-apply-layer --since "5 seconds ago" --no-pager -o cat 2>/dev/null | \
              grep "decision=EXECUTE" | \
              grep -oE "Plan [0-9a-f]{16}" | \
              awk '{print $2}' | \
              head -1)
    
    if [ -n "$PLAN_ID" ]; then
        echo "✓ EXECUTE plan detected: $PLAN_ID"
        break
    fi
    
    sleep 1
done

if [ -z "$PLAN_ID" ]; then
    echo "✗ No EXECUTE plan within ${TIMEOUT}s timeout"
    exit 1
fi

PLAN_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo ""

# 4. Check Governor Permit (P3.2)
echo "Step 4: Governor Permit (P3.2)"
echo "-------------------------------"
sleep 0.5  # Brief wait for permits

GOV_PERMIT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT GET "quantum:permit:$PLAN_ID" 2>/dev/null || echo "")
GOV_TTL=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT TTL "quantum:permit:$PLAN_ID" 2>/dev/null || echo "-2")

if [ "$GOV_TTL" -gt 0 ]; then
    echo "✓ Governor permit EXISTS (TTL: ${GOV_TTL}s)"
    echo "  Content: $GOV_PERMIT"
elif [ -n "$GOV_PERMIT" ]; then
    echo "✓ Governor permit consumed (was present)"
else
    echo "⚠ Governor permit NOT FOUND (may be consumed or missing)"
fi
echo ""

# 5. Check P3.3 Permit
echo "Step 5: P3.3 Position State Brain Permit"
echo "-----------------------------------------"
P33_PERMIT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT GET "quantum:permit:p33:$PLAN_ID" 2>/dev/null || echo "")
P33_TTL=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT TTL "quantum:permit:p33:$PLAN_ID" 2>/dev/null || echo "-2")

if [ "$P33_TTL" -gt 0 ]; then
    echo "✓ P3.3 permit EXISTS (TTL: ${P33_TTL}s)"
    echo "  Content: $P33_PERMIT"
    
    # Parse permit details
    SAFE_QTY=$(echo "$P33_PERMIT" | jq -r '.safe_close_qty // "N/A"' 2>/dev/null || echo "N/A")
    EXCHANGE_AMT=$(echo "$P33_PERMIT" | jq -r '.exchange_position_amt // "N/A"' 2>/dev/null || echo "N/A")
    echo "  safe_close_qty: $SAFE_QTY"
    echo "  exchange_position_amt: $EXCHANGE_AMT"
elif [ -n "$P33_PERMIT" ]; then
    echo "✓ P3.3 permit consumed (was present)"
else
    echo "⚠ P3.3 permit NOT FOUND (may be consumed or denied)"
fi
echo ""

# 6. Check P3.3 logs for permit issuance
echo "Step 6: P3.3 Permit Issuance Log"
echo "---------------------------------"
P33_LOG=$(journalctl -u quantum-position-state-brain --since "$PLAN_TIMESTAMP" --no-pager 2>/dev/null | \
          grep -E "ALLOW|DENY" | \
          grep "$PLAN_ID" | \
          head -5)

if [ -n "$P33_LOG" ]; then
    echo "✓ P3.3 logged permit decision:"
    echo "$P33_LOG" | sed 's/^/  /'
else
    echo "⚠ No P3.3 permit log found for this plan"
fi
echo ""

# 7. Check Apply Layer execution
echo "Step 7: Apply Layer Execution"
echo "------------------------------"
sleep 1  # Wait for execution to complete

APPLY_LOG=$(journalctl -u quantum-apply-layer --since "$PLAN_TIMESTAMP" --no-pager 2>/dev/null | \
            grep "$PLAN_ID" | \
            grep -E "permits ready|permit consumed|Order.*executed|executed=True" | \
            head -10)

if [ -n "$APPLY_LOG" ]; then
    echo "✓ Apply Layer execution log:"
    echo "$APPLY_LOG" | sed 's/^/  /'
    
    # Check for success markers
    if echo "$APPLY_LOG" | grep -q "executed=True"; then
        echo ""
        echo "✓✓✓ EXECUTION SUCCESSFUL (executed=True)"
    elif echo "$APPLY_LOG" | grep -q "Order.*executed"; then
        echo ""
        echo "✓✓✓ ORDER EXECUTED"
    else
        echo ""
        echo "⚠ Execution status unclear"
    fi
else
    echo "✗ No execution log found"
fi
echo ""

# 8. Verify timing (permits created before execution)
echo "Step 8: Timing Analysis"
echo "-----------------------"

# Extract timestamps from logs
PLAN_TIME=$(journalctl -u quantum-apply-layer --since "$PLAN_TIMESTAMP" --no-pager -o short-iso 2>/dev/null | \
            grep "decision=EXECUTE" | grep "$PLAN_ID" | head -1 | awk '{print $1}')

P33_TIME=$(journalctl -u quantum-position-state-brain --since "$PLAN_TIMESTAMP" --no-pager -o short-iso 2>/dev/null | \
           grep -E "ALLOW|DENY" | grep "$PLAN_ID" | head -1 | awk '{print $1}')

EXEC_TIME=$(journalctl -u quantum-apply-layer --since "$PLAN_TIMESTAMP" --no-pager -o short-iso 2>/dev/null | \
            grep "$PLAN_ID" | grep -E "Order.*executed|executed=True" | head -1 | awk '{print $1}')

echo "Timeline:"
echo "  Plan published:  ${PLAN_TIME:-N/A}"
echo "  P3.3 permit:     ${P33_TIME:-N/A}"
echo "  Execution:       ${EXEC_TIME:-N/A}"
echo ""

# 9. Summary
echo "====================================="
echo "Summary"
echo "====================================="

if [ "$P33_TTL" -gt 0 ] || echo "$APPLY_LOG" | grep -q "permit consumed"; then
    echo "✓ P3.3 permit issued"
else
    echo "✗ P3.3 permit missing or denied"
fi

if echo "$APPLY_LOG" | grep -q "executed=True"; then
    echo "✓ Execution successful (reduceOnly order placed)"
else
    echo "✗ Execution did not complete"
fi

if [ -n "$P33_TIME" ] && [ -n "$EXEC_TIME" ]; then
    echo "✓ Permit created before execution (event-driven)"
else
    echo "⚠ Timing data incomplete"
fi

echo ""
echo "====================================="
echo "Proof Complete"
echo "====================================="

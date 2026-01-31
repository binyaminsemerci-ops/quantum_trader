#!/bin/bash
START_TIME=$(date +%s)
BASELINE_PENDING=481
EXECUTED_TRUE_COUNT=0
ORDER_ID_MISSING_COUNT=0
CONDITIONAL_BLOCKED_COUNT=0
FAIL_REASON=""

echo "=== PHASE 3 MONITORING START ==="
echo "Start time: $START_TIME"
echo "Baseline exit_intelligence pending: $BASELINE_PENDING"
echo "Window: 600 seconds (10 minutes)"
echo ""

SAMPLE_NUM=0
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -ge 600 ]; then
        echo "=== 10 MINUTE WINDOW COMPLETE ==="
        break
    fi
    
    SAMPLE_NUM=$((SAMPLE_NUM + 1))
    echo "--- Sample $SAMPLE_NUM (elapsed: ${ELAPSED}s) ---"
    
    # Check apply.result for SOLUSDT
    RECENT=$(/usr/bin/redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10)
    SOLUSDT_EXEC=$(echo "$RECENT" | grep -A10 "SOLUSDT" | grep "^executed$" -A1 | tail -1)
    
    if [ "$SOLUSDT_EXEC" = "True" ]; then
        EXECUTED_TRUE_COUNT=$((EXECUTED_TRUE_COUNT + 1))
        echo "✓ SOLUSDT executed=True detected (count: $EXECUTED_TRUE_COUNT)"
        
        # Check for order_id
        ORDER_ID=$(echo "$RECENT" | grep -A10 "SOLUSDT" | grep "^order_id$" -A1 | tail -1)
        if [ -z "$ORDER_ID" ] || [ "$ORDER_ID" = "" ]; then
            ORDER_ID_MISSING_COUNT=$((ORDER_ID_MISSING_COUNT + 1))
            FAIL_REASON="FAIL: executed=True without order_id"
            echo "$FAIL_REASON"
            break
        fi
    fi
    
    # Check gateway for conditional orders
    GATEWAY_LOGS=$(journalctl -u exit_order_gateway --since "1 min ago" --no-pager 2>/dev/null | egrep -i "BLOCKED|STOP|TAKE_PROFIT|TRAILING|conditional" | wc -l)
    if [ $GATEWAY_LOGS -gt 0 ]; then
        CONDITIONAL_BLOCKED_COUNT=$((CONDITIONAL_BLOCKED_COUNT + GATEWAY_LOGS))
        FAIL_REASON="FAIL: Conditional order attempt detected"
        echo "$FAIL_REASON"
        break
    fi
    
    # Check exit_intelligence pending
    CURRENT_PENDING=$(/usr/bin/redis-cli XINFO GROUPS quantum:stream:apply.result | grep -A6 "exit_intelligence" | grep "pending" -A1 | tail -1)
    PENDING_INCREASE=$((CURRENT_PENDING - BASELINE_PENDING))
    echo "exit_intelligence pending: $CURRENT_PENDING (change: $PENDING_INCREASE)"
    
    if [ $PENDING_INCREASE -gt 200 ]; then
        FAIL_REASON="FAIL: exit_intelligence pending increased by $PENDING_INCREASE (>200)"
        echo "$FAIL_REASON"
        break
    fi
    
    # Check stream liveness
    LAST_MSG=$(/usr/bin/redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 1 | head -1)
    LAST_TS=$(echo $LAST_MSG | cut -d'-' -f1)
    LAST_TS_SEC=$((LAST_TS / 1000))
    AGE=$((CURRENT_TIME - LAST_TS_SEC))
    
    echo "apply.result age: ${AGE}s"
    if [ $AGE -gt 60 ]; then
        FAIL_REASON="FAIL: apply.result not updating (age: ${AGE}s)"
        echo "$FAIL_REASON"
        break
    fi
    
    # Check Apply Layer status
    AL_STATUS=$(systemctl is-active quantum-apply-layer)
    if [ "$AL_STATUS" != "active" ]; then
        FAIL_REASON="FAIL: Apply Layer not active (status: $AL_STATUS)"
        echo "$FAIL_REASON"
        break
    fi
    
    echo ""
    sleep 30
done

echo ""
echo "=== MONITORING SUMMARY ==="
echo "EXECUTED_TRUE_COUNT=$EXECUTED_TRUE_COUNT"
echo "ORDER_ID_MISSING_COUNT=$ORDER_ID_MISSING_COUNT"
echo "CONDITIONAL_BLOCKED_COUNT=$CONDITIONAL_BLOCKED_COUNT"
if [ -n "$FAIL_REASON" ]; then
    echo "RESULT=FAIL"
    echo "REASON=$FAIL_REASON"
else
    echo "RESULT=PASS"
fi

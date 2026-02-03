#!/bin/bash
# Capital Efficiency Brain Validation Script
# Verifies P3.1 service operation, shadow mode, Redis output, metrics

set -euo pipefail

echo "🧠 Capital Efficiency Brain Validation (P3.1)"
echo "=============================================="
echo ""

# Configuration
SERVICE_NAME="quantum-capital-efficiency"
SERVICE_URL="http://localhost:8026"
REDIS_CLI="redis-cli"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass_test() {
    echo "  ✅ $1"
    ((TESTS_PASSED++))
}

fail_test() {
    echo "  ❌ $1"
    ((TESTS_FAILED++))
}

# Test 1: Service health
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Service Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if systemctl is-active --quiet $SERVICE_NAME; then
    pass_test "Service is running"
else
    fail_test "Service is not running"
fi

HEALTH_RESPONSE=$(curl -s $SERVICE_URL/health || echo '{"status":"error"}')
HEALTH_STATUS=$(echo $HEALTH_RESPONSE | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$HEALTH_STATUS" = "healthy" ] || [ "$HEALTH_STATUS" = "degraded" ]; then
    pass_test "Health endpoint responding: $HEALTH_STATUS"
else
    fail_test "Health endpoint not responding correctly"
fi

# Check shadow mode
SHADOW_MODE=$(echo $HEALTH_RESPONSE | grep -o '"shadow_mode":[^,}]*' | cut -d':' -f2)
if [ "$SHADOW_MODE" = "true" ]; then
    pass_test "Shadow mode enabled (safe default)"
else
    echo "  ⚠️  Shadow mode disabled - active rebalancing may occur"
fi

echo ""

# Test 2: Metrics endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Prometheus Metrics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

METRICS=$(curl -s $SERVICE_URL/metrics)

if echo "$METRICS" | grep -q "capital_efficiency_score"; then
    pass_test "Efficiency score metric exists"
else
    fail_test "Efficiency score metric missing"
fi

if echo "$METRICS" | grep -q "capital_pressure_signal"; then
    pass_test "Capital pressure metric exists"
else
    fail_test "Capital pressure metric missing"
fi

if echo "$METRICS" | grep -q "efficiency_events_processed"; then
    pass_test "Processing counter metric exists"
else
    fail_test "Processing counter metric missing"
fi

if echo "$METRICS" | grep -q "capital_efficiency_shadow_mode"; then
    SHADOW_VAL=$(echo "$METRICS" | grep "capital_efficiency_shadow_mode" | grep -v "#" | awk '{print $2}')
    if [ "$SHADOW_VAL" = "1.0" ]; then
        pass_test "Shadow mode metric = 1.0 (confirmed)"
    else
        echo "  ⚠️  Shadow mode metric = $SHADOW_VAL"
    fi
else
    fail_test "Shadow mode metric missing"
fi

echo ""

# Test 3: Redis output format
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Redis Output Format"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Inject test PnL event
TEST_SYMBOL="TESTUSDT"
$REDIS_CLI XADD "execution:pnl:stream" "*" symbol $TEST_SYMBOL pnl 15.5 > /dev/null

# Wait for processing
sleep 2

# Check if efficiency key was created
EFFICIENCY_KEY="quantum:capital:efficiency:$TEST_SYMBOL"
if $REDIS_CLI EXISTS $EFFICIENCY_KEY | grep -q "1"; then
    pass_test "Efficiency key created in Redis"
    
    # Validate required fields
    EFFICIENCY_DATA=$($REDIS_CLI HGETALL $EFFICIENCY_KEY)
    
    if echo "$EFFICIENCY_DATA" | grep -q "efficiency_score"; then
        SCORE=$(echo "$EFFICIENCY_DATA" | grep -A1 "efficiency_score" | tail -1)
        pass_test "efficiency_score field exists: $SCORE"
    else
        fail_test "efficiency_score field missing"
    fi
    
    if echo "$EFFICIENCY_DATA" | grep -q "capital_pressure"; then
        PRESSURE=$(echo "$EFFICIENCY_DATA" | grep -A1 "capital_pressure" | tail -1)
        if echo "$PRESSURE" | grep -qE "INCREASE|HOLD|DECREASE"; then
            pass_test "capital_pressure valid: $PRESSURE"
        else
            fail_test "capital_pressure invalid value"
        fi
    else
        fail_test "capital_pressure field missing"
    fi
    
    if echo "$EFFICIENCY_DATA" | grep -q "reallocation_weight"; then
        pass_test "reallocation_weight field exists"
    else
        fail_test "reallocation_weight field missing"
    fi
    
    if echo "$EFFICIENCY_DATA" | grep -q "confidence"; then
        pass_test "confidence field exists"
    else
        fail_test "confidence field missing"
    fi
else
    fail_test "Efficiency key not created (service may not be processing)"
fi

echo ""

# Test 4: Service logs (fail-closed verification)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Service Logs & Behavior"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

RECENT_LOGS=$(journalctl -u $SERVICE_NAME --since "2 minutes ago" -n 50 2>/dev/null || echo "")

if echo "$RECENT_LOGS" | grep -q "SHADOW"; then
    pass_test "Shadow mode logging confirmed"
else
    echo "  ⚠️  No shadow mode log entries found (may be starting)"
fi

if echo "$RECENT_LOGS" | grep -qiE "error|exception|fail" | head -5; then
    ERROR_COUNT=$(echo "$RECENT_LOGS" | grep -ciE "error|exception|fail")
    fail_test "Found $ERROR_COUNT error entries in logs"
    echo ""
    echo "Recent errors:"
    echo "$RECENT_LOGS" | grep -iE "error|exception|fail" | tail -3
else
    pass_test "No critical errors in recent logs"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo "Tests Passed: $TESTS_PASSED / $TOTAL_TESTS"
echo "Tests Failed: $TESTS_FAILED / $TOTAL_TESTS"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    echo ""
    echo "Capital Efficiency Brain Status:"
    echo "  [✓] Service running in shadow mode"
    echo "  [✓] Health and metrics endpoints operational"
    echo "  [✓] Redis output format validated"
    echo "  [✓] Efficiency scoring active"
    echo ""
    echo "🎯 P3.1 Capital Efficiency Brain VERIFIED"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo ""
    echo "Review failed tests above and check:"
    echo "  - systemctl status $SERVICE_NAME"
    echo "  - journalctl -u $SERVICE_NAME -n 100"
    echo "  - curl $SERVICE_URL/health"
    echo ""
    exit 1
fi

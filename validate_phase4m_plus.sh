#!/bin/bash
# PHASE 4M+ VALIDATION SCRIPT (LINUX/VPS VERSION)
# Validates Cross-Exchange Intelligence → ExitBrain v3 Integration

echo "======================================================================"
echo "PHASE 4M+ VALIDATION - Cross-Exchange → ExitBrain v3 Integration"
echo "======================================================================"
echo ""

ERROR_COUNT=0
WARNING_COUNT=0
TEST_COUNT=0

test_step() {
    local name="$1"
    local command="$2"
    local validation="$3"
    
    ((TEST_COUNT++))
    echo -n "[$TEST_COUNT] $name... "
    
    result=$(eval "$command" 2>&1)
    exit_code=$?
    
    if eval "$validation"; then
        echo "✅ PASS"
        return 0
    else
        echo "❌ FAIL"
        ((ERROR_COUNT++))
        [ -n "$result" ] && echo "    Output: $result"
        return 1
    fi
}

# ============================================================================
# TEST 1: Cross-Exchange Data Stream
# ============================================================================
echo -e "\n[CROSS-EXCHANGE DATA VALIDATION]"
echo "----------------------------------------------------------------------"

test_step "Check quantum:stream:exchange.raw exists" \
    "docker exec quantum_redis redis-cli EXISTS quantum:stream:exchange.raw" \
    "[ \"\$result\" = \"1\" ]"

test_step "Check exchange.raw has data (> 100 entries)" \
    "docker exec quantum_redis redis-cli XLEN quantum:stream:exchange.raw" \
    "[ \$result -gt 100 ]"

test_step "Check quantum:stream:exchange.normalized stream created" \
    "docker exec quantum_redis redis-cli EXISTS quantum:stream:exchange.normalized" \
    "[ \"\$result\" = \"1\" ]"

# ============================================================================
# TEST 2: ExitBrain Status Stream
# ============================================================================
echo -e "\n[EXITBRAIN STATUS STREAM]"
echo "----------------------------------------------------------------------"

test_step "Check quantum:stream:exitbrain.status stream exists" \
    "docker exec quantum_redis redis-cli EXISTS quantum:stream:exitbrain.status" \
    "[ \"\$result\" = \"1\" ]"

test_step "Check exitbrain.status has recent data" \
    "docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.status" \
    "[ \$result -gt 0 ]"

# ============================================================================
# TEST 3: Cross-Exchange Adapter Health
# ============================================================================
echo -e "\n[CROSS-EXCHANGE ADAPTER HEALTH]"
echo "----------------------------------------------------------------------"

test_step "Check AI Engine /health endpoint" \
    "curl -s http://localhost:8001/health | grep -c cross_exchange" \
    "[ \$result -gt 0 ]"

# ============================================================================
# TEST 4: Cross-Exchange State Verification
# ============================================================================
echo -e "\n[CROSS-EXCHANGE STATE]"
echo "----------------------------------------------------------------------"

latest_state=$(docker exec quantum_redis redis-cli --raw XREVRANGE quantum:stream:exchange.normalized + - COUNT 1)

if [ -n "$latest_state" ]; then
    echo "✓ Latest normalized data:"
    echo "$latest_state"
else
    echo "⚠️  No normalized stream data yet (aggregator may not be running)"
    ((WARNING_COUNT++))
fi

# ============================================================================
# TEST 5: ExitBrain Integration Logs
# ============================================================================
echo -e "\n[EXITBRAIN INTEGRATION LOGS]"
echo "----------------------------------------------------------------------"

test_step "Check for cross-exchange adapter initialization" \
    "docker logs quantum_ai_engine 2>&1 | grep -c 'Cross-Exchange'" \
    "[ \$result -gt 0 ]"

test_step "Check for adjustments in logs" \
    "docker logs quantum_ai_engine 2>&1 | grep -c 'adjustments applied\|volatility_factor'" \
    "[ \$result -gt 0 ]"

# ============================================================================
# TEST 6: Alert Stream (Fail-Safe Monitoring)
# ============================================================================
echo -e "\n[FAIL-SAFE MONITORING]"
echo "----------------------------------------------------------------------"

alerts=$(docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.alerts)

if [ "$alerts" = "0" ]; then
    echo "✓ No fallback alerts (system operating normally)"
else
    echo "⚠️  Found $alerts alerts, checking recent:"
    docker exec quantum_redis redis-cli --raw XREVRANGE quantum:stream:exitbrain.alerts + - COUNT 3
    ((WARNING_COUNT++))
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "======================================================================"
echo "VALIDATION SUMMARY"
echo "======================================================================"

echo -e "\nTests Run: $TEST_COUNT"
echo "Errors: $ERROR_COUNT"
echo "Warnings: $WARNING_COUNT"

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "\n✅ PHASE 4M+ INTEGRATION VALIDATED"
    echo "   Cross-Exchange Intelligence → ExitBrain v3 is operational"
    exit 0
else
    echo -e "\n❌ VALIDATION FAILED"
    echo "   Fix errors and re-run validation"
    exit 1
fi

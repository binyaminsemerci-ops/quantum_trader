#!/usr/bin/env bash
# P1-B: Logging Stack Status Verification
# Usage: bash scripts/log_status.sh

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "P1-B: LOGGING STACK STATUS"
echo "========================================"
echo ""

# Counter for passed/failed checks
PASS=0
FAIL=0

# Helper functions
pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASS++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAIL++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

# Check 1: Loki Status
echo "🔍 Checking Loki..."
if docker ps | grep -q quantum_loki; then
    # Check if running
    if curl -s --max-time 5 http://localhost:3100/ready | grep -q "ready"; then
        pass "Loki is UP and ready"
    else
        fail "Loki container running but not ready"
    fi
else
    fail "Loki container is not running"
fi
echo ""

# Check 2: Promtail Status
echo "🔍 Checking Promtail..."
if docker ps | grep -q quantum_promtail; then
    pass "Promtail is running"
    # Check if Promtail can reach Loki
    if docker exec quantum_promtail curl -s --max-time 5 http://loki:3100/ready | grep -q "ready"; then
        pass "Promtail → Loki connectivity OK"
    else
        fail "Promtail cannot reach Loki"
    fi
else
    fail "Promtail container is not running"
fi
echo ""

# Check 3: Grafana Loki Datasource
echo "🔍 Checking Grafana Loki Datasource..."
if curl -s --max-time 5 "http://localhost:3001/api/datasources/name/Loki" \
    -u admin:${GRAFANA_PASSWORD:-admin} 2>/dev/null | grep -q '"type":"loki"'; then
    pass "Grafana Loki datasource configured"
else
    fail "Grafana Loki datasource not found or misconfigured"
fi
echo ""

# Check 4: JSON Logging Active
echo "🔍 Checking JSON Logging Format..."
# Check auto_executor logs for JSON format
if docker logs quantum_auto_executor --tail 10 2>&1 | grep -q '"ts":.*"level":.*"service"'; then
    pass "Auto-executor is logging in JSON format"
else
    fail "Auto-executor is NOT logging in JSON format"
fi

# Check ai_engine logs for JSON format
if docker logs quantum_ai_engine --tail 10 2>&1 | grep -q '"ts":.*"level":.*"service"'; then
    pass "AI Engine is logging in JSON format"
else
    warn "AI Engine logs not in JSON format (may not have started yet)"
fi
echo ""

# Check 5: correlation_id Tracking
echo "🔍 Checking correlation_id Tracking..."
# Look for correlation_id in recent logs
if docker logs quantum_auto_executor --tail 50 2>&1 | grep -q '"correlation_id"'; then
    pass "correlation_id present in logs"
    
    # Try to find a full order flow with same correlation_id
    echo "   Checking order flow tracking..."
    CORR_ID=$(docker logs quantum_auto_executor --tail 100 2>&1 | grep -o '"correlation_id":"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ -n "$CORR_ID" ]; then
        echo "   Sample correlation_id: $CORR_ID"
        
        # Check if we can track across events
        ORDER_SUBMIT=$(docker logs quantum_auto_executor 2>&1 | grep "$CORR_ID" | grep -c "ORDER_SUBMIT" || true)
        ORDER_RESPONSE=$(docker logs quantum_auto_executor 2>&1 | grep "$CORR_ID" | grep -c "ORDER_RESPONSE" || true)
        
        if [ "$ORDER_SUBMIT" -gt 0 ] && [ "$ORDER_RESPONSE" -gt 0 ]; then
            pass "correlation_id tracks full order flow (SUBMIT → RESPONSE)"
        elif [ "$ORDER_SUBMIT" -gt 0 ]; then
            warn "correlation_id tracks ORDER_SUBMIT but no RESPONSE yet"
        else
            warn "No complete order flow found (may not have traded yet)"
        fi
    fi
else
    fail "correlation_id NOT found in logs"
fi
echo ""

# Check 6: Loki Data Ingestion
echo "🔍 Checking Loki Data Ingestion..."
# Check if Loki has any data
LABELS=$(curl -s --max-time 5 "http://localhost:3100/loki/api/v1/label/container/values" 2>/dev/null || echo "[]")
if echo "$LABELS" | grep -q "quantum_"; then
    pass "Loki is ingesting logs from Quantum containers"
    echo "   Containers: $(echo $LABELS | grep -o 'quantum_[^"]*' | tr '\n' ' ')"
else
    fail "Loki has no data from Quantum containers"
fi
echo ""

# Check 7: Alert Rules Loaded
echo "🔍 Checking Alert Rules..."
if curl -s --max-time 5 "http://localhost:9090/api/v1/rules" | grep -q "p1b_"; then
    pass "P1-B alert rules are loaded in Prometheus"
else
    fail "P1-B alert rules NOT loaded in Prometheus"
fi
echo ""

# Check 8: Alertmanager Routing
echo "🔍 Checking Alertmanager Configuration..."
if docker exec quantum_alertmanager cat /etc/alertmanager/alertmanager.yml | grep -q "critical-alerts"; then
    pass "Alertmanager has critical routing configured"
else
    fail "Alertmanager critical routing NOT configured"
fi
echo ""

# Check 9: Disk Space (Loki data)
echo "🔍 Checking Disk Space..."
if command -v df &> /dev/null; then
    DISK_USAGE=$(df -h | grep '/$' | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -lt 85 ]; then
        pass "Disk usage is ${DISK_USAGE}% (healthy)"
    elif [ "$DISK_USAGE" -lt 95 ]; then
        warn "Disk usage is ${DISK_USAGE}% (getting full)"
    else
        fail "Disk usage is ${DISK_USAGE}% (CRITICAL - Loki may fail)"
    fi
else
    warn "Cannot check disk space (df command not available)"
fi
echo ""

# Check 10: Runbooks Exist
echo "🔍 Checking Runbooks..."
if [ -f "RUNBOOKS/P0_execution_stuck.md" ] && [ -f "RUNBOOKS/P1B_logging_stack.md" ] && [ -f "RUNBOOKS/alerts.md" ]; then
    pass "All required runbooks exist"
else
    fail "Some runbooks are missing"
fi
echo ""

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo -e "${GREEN}Passed${NC}: $PASS"
echo -e "${RED}Failed${NC}: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
    echo "P1-B Logging Stack is operational"
    exit 0
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo "Review failures above and consult RUNBOOKS/P1B_logging_stack.md"
    exit 1
fi

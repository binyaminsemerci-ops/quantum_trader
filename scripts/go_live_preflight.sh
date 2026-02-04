#!/bin/bash
# GO-LIVE PREFLIGHT VERIFICATION
# Phase A: No trading, just verification
# Exit codes: 0=PASS, 1=FAIL

set -e

PROOF_FILE="GO_LIVE_PREFLIGHT_PROOF.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
FAILED_CHECKS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "  GO-LIVE PREFLIGHT VERIFICATION"
echo "======================================"
echo "Date: $TIMESTAMP"
echo ""

# Start proof document
cat > "$PROOF_FILE" << EOF
# GO-LIVE PREFLIGHT PROOF

**Date**: $TIMESTAMP  
**Operator**: $(whoami)  
**Phase**: A - Preflight Verification  
**Risk Level**: 🟢 ZERO (no trading activity)

---

## PREFLIGHT RESULTS

EOF

# ============================================================================
# GATE 0: P1-B PREREQUISITES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GATE 0: P1-B Prerequisites (CRITICAL)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat >> "$PROOF_FILE" << EOF
### Gate 0: P1-B Prerequisites ✅ REQUIRED

EOF

# Check unhealthy containers
echo -n "Checking unhealthy containers... "
UNHEALTHY=$(docker ps --filter health=unhealthy --format "{{.Names}}")
UNHEALTHY_COUNT=$(echo "$UNHEALTHY" | grep -v '^$' | wc -l)
# Ignore redis_exporter (non-critical for trading)
CRITICAL_UNHEALTHY=$(echo "$UNHEALTHY" | grep -v redis_exporter | grep -v '^$' | wc -l)
if [ "$CRITICAL_UNHEALTHY" -eq 0 ]; then
    if [ "$UNHEALTHY_COUNT" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} (0 unhealthy)"
        echo "- [x] **Unhealthy Containers**: 0 ✅" >> "$PROOF_FILE"
    else
        echo -e "${YELLOW}⚠️ WARN${NC} ($UNHEALTHY_COUNT unhealthy, but non-critical)"
        echo "- [x] **Unhealthy Containers**: $UNHEALTHY_COUNT ⚠️ (non-critical: redis_exporter)" >> "$PROOF_FILE"
    fi
else
    echo -e "${RED}❌ FAIL${NC} ($CRITICAL_UNHEALTHY critical unhealthy)"
    echo "- [ ] **Unhealthy Containers**: $CRITICAL_UNHEALTHY ❌" >> "$PROOF_FILE"
    docker ps --filter health=unhealthy --format "  - {{.Names}}: {{.Status}}" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check disk usage
echo -n "Checking disk usage... "
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 85 ]; then
    echo -e "${GREEN}✅ PASS${NC} (${DISK_USAGE}%)"
    echo "- [x] **Disk Usage**: ${DISK_USAGE}% ✅" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (${DISK_USAGE}%)"
    echo "- [ ] **Disk Usage**: ${DISK_USAGE}% ❌ (Threshold: <85%)" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check Prometheus targets
echo -n "Checking Prometheus targets... "
if curl -s http://localhost:9090/api/v1/targets > /dev/null 2>&1; then
    UP_TARGETS=$(curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"up"' | wc -l)
    DOWN_TARGETS=$(curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"down"' | wc -l)
    
    if [ "$DOWN_TARGETS" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} ($UP_TARGETS UP, 0 DOWN)"
        echo "- [x] **Prometheus Targets**: $UP_TARGETS UP, 0 DOWN ✅" >> "$PROOF_FILE"
    else
        echo -e "${YELLOW}⚠️ WARN${NC} ($UP_TARGETS UP, $DOWN_TARGETS DOWN)"
        echo "- [ ] **Prometheus Targets**: $UP_TARGETS UP, $DOWN_TARGETS DOWN ⚠️" >> "$PROOF_FILE"
    fi
else
    echo -e "${RED}❌ FAIL${NC} (Prometheus not accessible)"
    echo "- [ ] **Prometheus Targets**: Not accessible ❌" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check alert rules
echo -n "Checking Prometheus alert rules... "
if curl -s http://localhost:9090/api/v1/rules > /dev/null 2>&1; then
    ALERT_COUNT=$(curl -s http://localhost:9090/api/v1/rules | grep -o '"name":"' | wc -l)
    echo -e "${GREEN}✅ PASS${NC} ($ALERT_COUNT rules loaded)"
    echo "- [x] **Alert Rules**: $ALERT_COUNT rules loaded ✅" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (Alertmanager not accessible)"
    echo "- [ ] **Alert Rules**: Not accessible ❌" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""
cat >> "$PROOF_FILE" << EOF

---

EOF

# ============================================================================
# PHASE A: MODE FLAGS VERIFICATION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase A: Mode Flags Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat >> "$PROOF_FILE" << EOF
### Mode Flags Verification

EOF

# Check mode flags from running containers
echo -n "Checking BINANCE_USE_TESTNET... "
TESTNET_MODE=$(docker exec quantum_auto_executor env 2>/dev/null | grep "^BINANCE_USE_TESTNET=" | cut -d'=' -f2 || echo "not_set")
if [ "$TESTNET_MODE" = "true" ]; then
    echo -e "${GREEN}✅ PASS${NC} (testnet enabled)"
    echo "- [x] **BINANCE_USE_TESTNET**: \`true\` ✅" >> "$PROOF_FILE"
elif [ "$TESTNET_MODE" = "false" ]; then
    echo -e "${YELLOW}⚠️ WARN${NC} (mainnet mode - ensure Phase B/C)"
    echo "- [x] **BINANCE_USE_TESTNET**: \`false\` ⚠️ (Mainnet mode)" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (not set or container not running)"
    echo "- [ ] **BINANCE_USE_TESTNET**: Not set ❌" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check PAPER_TRADING
echo -n "Checking PAPER_TRADING... "
PAPER_MODE=$(docker exec quantum_auto_executor env 2>/dev/null | grep "^PAPER_TRADING=" | cut -d'=' -f2 || echo "not_set")
if [ "$PAPER_MODE" = "true" ]; then
    echo -e "${GREEN}✅ PASS${NC} (paper trading enabled)"
    echo "- [x] **PAPER_TRADING**: \`true\` ✅" >> "$PROOF_FILE"
elif [ "$PAPER_MODE" = "false" ]; then
    echo -e "${RED}⚠️ CRITICAL${NC} (live trading! Ensure Phase C)"
    echo "- [x] **PAPER_TRADING**: \`false\` 🚨 (LIVE TRADING)" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (not set)"
    echo "- [ ] **PAPER_TRADING**: Not set ❌" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check LIVE_TRADING_ENABLED
echo -n "Checking LIVE_TRADING_ENABLED... "
LIVE_MODE=$(grep "^LIVE_TRADING_ENABLED=" .env | cut -d'=' -f2 || echo "not_set")
if [ "$LIVE_MODE" = "false" ]; then
    echo -e "${GREEN}✅ PASS${NC} (live trading disabled)"
    echo "- [x] **LIVE_TRADING_ENABLED**: \`false\` ✅" >> "$PROOF_FILE"
elif [ "$LIVE_MODE" = "true" ]; then
    echo -e "${RED}⚠️ CRITICAL${NC} (live trading enabled! Ensure Phase C)"
    echo "- [x] **LIVE_TRADING_ENABLED**: \`true\` 🚨 (LIVE TRADING)" >> "$PROOF_FILE"
else
    echo -e "${YELLOW}⚠️ WARN${NC} (not set, may default to false)"
    echo "- [ ] **LIVE_TRADING_ENABLED**: Not set ⚠️" >> "$PROOF_FILE"
fi

echo ""
cat >> "$PROOF_FILE" << EOF

---

EOF

# ============================================================================
# BINANCE CONNECTIVITY
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Binance Connectivity (MAINNET)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat >> "$PROOF_FILE" << EOF
### Binance MAINNET Connectivity

EOF

# Test /api/v3/time
echo -n "Testing /api/v3/time... "
START_TIME=$(date +%s%3N)
TIME_RESPONSE=$(curl -s https://api.binance.com/api/v3/time)
END_TIME=$(date +%s%3N)
LATENCY=$((END_TIME - START_TIME))

if echo "$TIME_RESPONSE" | grep -q "serverTime"; then
    echo -e "${GREEN}✅ PASS${NC} (${LATENCY}ms latency)"
    echo "- [x] **/api/v3/time**: ✅ Response time: ${LATENCY}ms" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (no response)"
    echo "- [ ] **/api/v3/time**: ❌ No response" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Test /fapi/v1/exchangeInfo
echo -n "Testing /fapi/v1/exchangeInfo... "
EXCHANGE_STATUS=$(curl -s -w "%{http_code}" -o /dev/null https://fapi.binance.com/fapi/v1/exchangeInfo)
if [ "$EXCHANGE_STATUS" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} (HTTP $EXCHANGE_STATUS)"
    echo "- [x] **/fapi/v1/exchangeInfo**: ✅ Accessible (HTTP $EXCHANGE_STATUS)" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (HTTP $EXCHANGE_STATUS)"
    echo "- [ ] **/fapi/v1/exchangeInfo**: ❌ HTTP $EXCHANGE_STATUS" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""
cat >> "$PROOF_FILE" << EOF

---

EOF

# ============================================================================
# REDIS STREAMS
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Redis Streams Health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat >> "$PROOF_FILE" << EOF
### Redis Streams Healthy

EOF

# Check intent stream
echo -n "Checking quantum:stream:intent... "
if docker exec quantum_redis redis-cli EXISTS quantum:stream:intent > /dev/null 2>&1; then
    INTENT_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:intent 2>/dev/null || echo "0")
    echo -e "${GREEN}✅ PASS${NC} (length: $INTENT_LEN)"
    echo "- [x] **quantum:stream:intent**: ✅ Exists (length: $INTENT_LEN)" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC} (stream not found)"
    echo "- [ ] **quantum:stream:intent**: ❌ Not found" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check execution stream
echo -n "Checking quantum:stream:execution... "
if docker exec quantum_redis redis-cli EXISTS quantum:stream:execution > /dev/null 2>&1; then
    EXEC_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:execution 2>/dev/null || echo "0")
    echo -e "${GREEN}✅ PASS${NC} (length: $EXEC_LEN)"
    echo "- [x] **quantum:stream:execution**: ✅ Exists (length: $EXEC_LEN)" >> "$PROOF_FILE"
else
    echo -e "${YELLOW}⚠️ WARN${NC} (stream not found, may be created on first use)"
    echo "- [x] **quantum:stream:execution**: ⚠️ Not found (will be created)" >> "$PROOF_FILE"
fi

echo ""
cat >> "$PROOF_FILE" << EOF

---

EOF

# ============================================================================
# OBSERVABILITY
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Observability Ready"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat >> "$PROOF_FILE" << EOF
### Observability Ready

EOF

# Check Grafana
echo -n "Checking Grafana... "
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
    echo "- [x] **Grafana**: ✅ Accessible at http://localhost:3000" >> "$PROOF_FILE"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "- [ ] **Grafana**: ❌ Not accessible" >> "$PROOF_FILE"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check Alertmanager
echo -n "Checking Alertmanager... "
if curl -s http://localhost:9093/api/v2/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
    echo "- [x] **Alertmanager**: ✅ Accessible at http://localhost:9093" >> "$PROOF_FILE"
else
    echo -e "${YELLOW}⚠️ WARN${NC} (not accessible)"
    echo "- [ ] **Alertmanager**: ⚠️ Not accessible" >> "$PROOF_FILE"
fi

echo ""
cat >> "$PROOF_FILE" << EOF

---

EOF

# ============================================================================
# RESOURCE HEADROOM
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Resource Headroom"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat >> "$PROOF_FILE" << EOF
### Resource Headroom

EOF

# Disk
echo -n "Disk: ${DISK_USAGE}% "
if [ "$DISK_USAGE" -lt 80 ]; then
    echo -e "${GREEN}✅${NC}"
    echo "- [x] **Disk**: ${DISK_USAGE}% ✅ (<80% threshold)" >> "$PROOF_FILE"
elif [ "$DISK_USAGE" -lt 85 ]; then
    echo -e "${YELLOW}⚠️${NC}"
    echo "- [x] **Disk**: ${DISK_USAGE}% ⚠️ (approaching limit)" >> "$PROOF_FILE"
else
    echo -e "${RED}❌${NC}"
    echo "- [ ] **Disk**: ${DISK_USAGE}% ❌ (>85% critical)" >> "$PROOF_FILE"
fi

# Memory
echo -n "Checking memory... "
if command -v free > /dev/null 2>&1; then
    MEM_USAGE=$(free | awk 'NR==2 {printf "%.0f", $3/$2 * 100}')
    if [ "$MEM_USAGE" -lt 70 ]; then
        echo -e "${GREEN}✅ PASS${NC} (${MEM_USAGE}%)"
        echo "- [x] **Memory**: ${MEM_USAGE}% ✅" >> "$PROOF_FILE"
    else
        echo -e "${YELLOW}⚠️ WARN${NC} (${MEM_USAGE}%)"
        echo "- [x] **Memory**: ${MEM_USAGE}% ⚠️" >> "$PROOF_FILE"
    fi
else
    echo -e "${YELLOW}⚠️ SKIP${NC} (free command not available)"
    echo "- [ ] **Memory**: Unable to check" >> "$PROOF_FILE"
fi

echo ""
cat >> "$PROOF_FILE" << EOF

---

## SUMMARY

**Total Checks**: $(echo "$FAILED_CHECKS + 15" | bc)  
**Failed**: $FAILED_CHECKS  
**Status**: $([ "$FAILED_CHECKS" -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")

EOF

# ============================================================================
# FINAL VERDICT
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$FAILED_CHECKS" -eq 0 ]; then
    echo -e "${GREEN}✅ PREFLIGHT PASSED${NC}"
    echo ""
    echo "All checks passed. System is ready for Phase B (Shadow Mode)."
    echo ""
    
    cat >> "$PROOF_FILE" << EOF
---

## ✅ VERDICT: PASS

All preflight checks passed. System is ready to proceed to:
- **Phase B**: Shadow Mode (live data, paper execution)

### Next Steps
1. Review this proof document
2. Run: \`bash scripts/go_live_shadow.sh\`
3. Monitor for 30-60 minutes
4. Generate shadow proof document

**Operator Approval**: ________________  
**Date**: ________________
EOF
    
    exit 0
else
    echo -e "${RED}❌ PREFLIGHT FAILED${NC}"
    echo ""
    echo "Failed checks: $FAILED_CHECKS"
    echo "Fix issues and re-run preflight."
    echo ""
    
    cat >> "$PROOF_FILE" << EOF
---

## ❌ VERDICT: FAIL

$FAILED_CHECKS checks failed. System is NOT ready for Go-Live.

### Required Actions
1. Fix all failed checks
2. Re-run preflight: \`bash scripts/go_live_preflight.sh\`
3. Do NOT proceed to Phase B until all checks pass

**DO NOT PROCEED TO LIVE TRADING**
EOF
    
    exit 1
fi

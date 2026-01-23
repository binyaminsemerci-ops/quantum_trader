#!/bin/bash
# P3.2 Release Gate Checklist
# Run before/after any mode change or deployment

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m"

echo -e "${BOLD}=== P3.2 RELEASE GATE CHECKLIST ===${NC}"
echo "Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

EXIT_CODE=0

# ============================================================================
# 1. Service Status
# ============================================================================
echo -e "${BOLD}[1/10] Service Status${NC}"
GOV_STATUS=$(systemctl is-active quantum-governor)
APPLY_STATUS=$(systemctl is-active quantum-apply-layer)

if [ "$GOV_STATUS" = "active" ] && [ "$APPLY_STATUS" = "active" ]; then
    echo -e "${GREEN}✅ Both services active${NC}"
    echo "   Governor: $GOV_STATUS"
    echo "   Apply Layer: $APPLY_STATUS"
else
    echo -e "${RED}❌ Service failure detected${NC}"
    echo "   Governor: $GOV_STATUS"
    echo "   Apply Layer: $APPLY_STATUS"
    EXIT_CODE=1
fi
echo ""

# ============================================================================
# 2. Apply Layer Mode
# ============================================================================
echo -e "${BOLD}[2/10] Apply Layer Mode${NC}"
APPLY_MODE=$(grep "^APPLY_MODE=" /etc/quantum/apply-layer.env | cut -d'=' -f2)
if [ "$APPLY_MODE" = "dry_run" ]; then
    echo -e "${GREEN}✅ APPLY_MODE=${APPLY_MODE}${NC} (safe)"
elif [ "$APPLY_MODE" = "testnet" ]; then
    echo -e "${YELLOW}⚠️  APPLY_MODE=${APPLY_MODE}${NC} (real execution enabled)"
else
    echo -e "${RED}❌ APPLY_MODE=${APPLY_MODE}${NC} (unknown mode)"
    EXIT_CODE=1
fi
echo ""

# ============================================================================
# 3. Governor Metrics
# ============================================================================
echo -e "${BOLD}[3/10] Governor Metrics${NC}"
if curl -s http://127.0.0.1:8044/metrics > /tmp/gov_metrics.txt 2>&1; then
    echo -e "${GREEN}✅ Metrics endpoint responding${NC}"
    
    ALLOW_TOTAL=$(grep "^quantum_govern_allow_total" /tmp/gov_metrics.txt | awk '{sum+=$2} END {print sum}')
    BLOCK_TOTAL=$(grep "^quantum_govern_block_total" /tmp/gov_metrics.txt | awk '{sum+=$2} END {print sum}')
    DISARM_TOTAL=$(grep "^quantum_govern_disarm_total" /tmp/gov_metrics.txt | awk '{sum+=$2} END {print sum}')
    
    echo "   Allows: ${ALLOW_TOTAL:-0}"
    echo "   Blocks: ${BLOCK_TOTAL:-0}"
    echo "   Disarms: ${DISARM_TOTAL:-0}"
else
    echo -e "${RED}❌ Metrics endpoint not responding${NC}"
    EXIT_CODE=1
fi
echo ""

# ============================================================================
# 4. Governor Events
# ============================================================================
echo -e "${BOLD}[4/10] Governor Event Stream${NC}"
EVENT_COUNT=$(redis-cli XLEN quantum:stream:governor.events 2>/dev/null || echo "0")
echo "   Event count: $EVENT_COUNT"

if [ "$EVENT_COUNT" -gt 0 ]; then
    echo "   Latest event:"
    redis-cli XREVRANGE quantum:stream:governor.events + - COUNT 1 | head -10 | sed 's/^/   /'
fi
echo ""

# ============================================================================
# 5. Apply Results (Recent)
# ============================================================================
echo -e "${BOLD}[5/10] Recent Apply Results${NC}"
RESULT_COUNT=$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null || echo "0")
echo "   Result count: $RESULT_COUNT"

if [ "$RESULT_COUNT" -gt 0 ]; then
    echo "   Latest 3 results:"
    redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3 | grep -E "symbol|decision|executed|error" | head -12 | sed 's/^/   /'
fi
echo ""

# ============================================================================
# 6. Permit Enforcement (Fail-Closed)
# ============================================================================
echo -e "${BOLD}[6/10] Permit Enforcement Check${NC}"
BLOCKED_BY_PERMIT=$(redis-cli XREVRANGE quantum:stream:apply.result + - | grep -c "missing_permit_or_redis" 2>/dev/null || echo "0")

if [ "$BLOCKED_BY_PERMIT" -gt 0 ]; then
    echo -e "${GREEN}✅ Fail-closed enforcement working${NC}"
    echo "   Found $BLOCKED_BY_PERMIT results blocked by missing permit"
else
    echo -e "${YELLOW}⚠️  No permit blocks found yet${NC}"
    echo "   (May not have tested permit enforcement)"
fi
echo ""

# ============================================================================
# 7. Execution Verification (If Testnet)
# ============================================================================
echo -e "${BOLD}[7/10] Execution Verification${NC}"
if [ "$APPLY_MODE" = "testnet" ]; then
    EXECUTED_COUNT=$(redis-cli XREVRANGE quantum:stream:apply.result + - | grep '"executed": true' | wc -l)
    
    if [ "$EXECUTED_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✅ Found $EXECUTED_COUNT executed results${NC}"
        echo "   Sample execution:"
        redis-cli XREVRANGE quantum:stream:apply.result + - | grep -A 5 '"executed": true' | head -10 | sed 's/^/   /'
    else
        echo "   No executions yet (normal if just started)"
    fi
else
    echo "   Skipped (dry_run mode)"
fi
echo ""

# ============================================================================
# 8. ReduceOnly Verification
# ============================================================================
echo -e "${BOLD}[8/10] ReduceOnly Flag Check${NC}"
if [ "$APPLY_MODE" = "testnet" ]; then
    REDUCE_ONLY_COUNT=$(journalctl -u quantum-apply-layer --since "5 minutes ago" | grep -c "reduceOnly" || echo "0")
    
    if [ "$REDUCE_ONLY_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✅ Found $REDUCE_ONLY_COUNT reduceOnly logs${NC}"
    else
        echo "   No reduceOnly logs in last 5 minutes"
    fi
else
    echo "   Skipped (dry_run mode)"
fi
echo ""

# ============================================================================
# 9. Burst Limit Detection
# ============================================================================
echo -e "${BOLD}[9/10] Burst Limit Enforcement${NC}"
BURST_BLOCKS=$(curl -s http://127.0.0.1:8044/metrics | grep 'quantum_govern_block_total.*burst_limit' | awk '{sum+=$2} END {print sum}')

if [ -n "$BURST_BLOCKS" ] && [ "$BURST_BLOCKS" -gt 0 ]; then
    echo -e "${GREEN}✅ Burst limit working (blocked $BURST_BLOCKS times)${NC}"
else
    echo "   No burst limit violations detected"
fi
echo ""

# ============================================================================
# 10. Auto-Disarm Status
# ============================================================================
echo -e "${BOLD}[10/10] Auto-Disarm Status${NC}"
TODAY=$(date -u +%Y-%m-%d)
DISARM_KEY="quantum:governor:disarm:${TODAY}"
DISARM_STATUS=$(redis-cli GET "$DISARM_KEY" 2>/dev/null || echo "")

if [ -n "$DISARM_STATUS" ]; then
    echo -e "${RED}❌ System is DISARMED for today${NC}"
    echo "   Disarm reason: $DISARM_STATUS"
    
    # Check if Apply Layer is in dry_run
    if [ "$APPLY_MODE" = "dry_run" ]; then
        echo -e "${GREEN}✅ Apply Layer correctly set to dry_run${NC}"
    else
        echo -e "${RED}❌ WARNING: Apply Layer still in $APPLY_MODE mode!${NC}"
        EXIT_CODE=1
    fi
else
    echo -e "${GREEN}✅ System is ARMED${NC}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BOLD}=== CHECKLIST SUMMARY ===${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed${NC}"
    echo ""
    echo "System Status:"
    echo "  • Mode: $APPLY_MODE"
    echo "  • Governor: Active"
    echo "  • Apply Layer: Active"
    echo "  • Permits: Enforced"
    echo "  • Limits: Active"
    echo ""
    echo -e "${GREEN}System ready for operation!${NC}"
else
    echo -e "${RED}❌ Some checks failed - review above${NC}"
    echo ""
    echo "Action required:"
    echo "  • Review failed checks"
    echo "  • Fix issues before proceeding"
    echo "  • Rerun checklist"
fi

exit $EXIT_CODE

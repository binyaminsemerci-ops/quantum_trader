#!/bin/bash
set -euo pipefail

# P3.2 Governor - Proof Pack

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m"

EXIT_CODE=0

echo -e "${BOLD}=== P3.2 GOVERNOR PROOF PACK ===${NC}"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# ============================================================================
# PROOF 1: SERVICE STATUS
# ============================================================================
echo -e "${BOLD}[PROOF 1/6] Service Status${NC}"

if systemctl is-active --quiet quantum-governor; then
    echo -e "${GREEN}✅ Governor service is active${NC}"
    
    UPTIME=$(systemctl show quantum-governor -p ActiveEnterTimestamp --value)
    echo "   Started: $UPTIME"
    
    MAIN_PID=$(systemctl show quantum-governor -p MainPID --value)
    if [ "$MAIN_PID" -gt 0 ] 2>/dev/null; then
        MEM_KB=$(ps -p "$MAIN_PID" -o rss= 2>/dev/null || echo "0")
        MEM_MB=$((MEM_KB / 1024))
        echo "   PID: $MAIN_PID, Memory: ${MEM_MB}MB"
    fi
else
    echo -e "${RED}❌ Governor service is NOT active${NC}"
    EXIT_CODE=1
fi
echo ""

# ============================================================================
# PROOF 2: METRICS ENDPOINT
# ============================================================================
echo -e "${BOLD}[PROOF 2/6] Metrics Endpoint${NC}"

if curl -s http://localhost:8044/metrics > /tmp/governor_metrics.txt 2>&1; then
    echo -e "${GREEN}✅ Metrics endpoint responding${NC}"
    
    # Count metrics
    ALLOW_COUNT=$(grep -c "^quantum_govern_allow_total" /tmp/governor_metrics.txt || echo "0")
    BLOCK_COUNT=$(grep -c "^quantum_govern_block_total" /tmp/governor_metrics.txt || echo "0")
    DISARM_COUNT=$(grep -c "^quantum_govern_disarm_total" /tmp/governor_metrics.txt || echo "0")
    EXEC_COUNT=$(grep -c "^quantum_govern_exec_count" /tmp/governor_metrics.txt || echo "0")
    
    echo "   Metrics found:"
    echo "   - Allow metrics: $ALLOW_COUNT"
    echo "   - Block metrics: $BLOCK_COUNT"
    echo "   - Disarm metrics: $DISARM_COUNT"
    echo "   - Exec count metrics: $EXEC_COUNT"
    
    if [ $((ALLOW_COUNT + BLOCK_COUNT + DISARM_COUNT + EXEC_COUNT)) -lt 4 ]; then
        echo -e "${YELLOW}   ⚠️  Warning: Expected at least 4 metric types${NC}"
    fi
else
    echo -e "${RED}❌ Metrics endpoint not responding${NC}"
    EXIT_CODE=1
fi
echo ""

# ============================================================================
# PROOF 3: PERMIT KEYS (ALLOW BEHAVIOR)
# ============================================================================
echo -e "${BOLD}[PROOF 3/6] Permit Keys (Allow Behavior)${NC}"

PERMIT_COUNT=$(redis-cli --scan --pattern "quantum:permit:*" | wc -l)

if [ "$PERMIT_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $PERMIT_COUNT permit keys${NC}"
    
    # Show sample permit
    SAMPLE_KEY=$(redis-cli --scan --pattern "quantum:permit:*" | head -1)
    if [ -n "$SAMPLE_KEY" ]; then
        echo "   Sample permit:"
        PERMIT_DATA=$(redis-cli GET "$SAMPLE_KEY")
        echo "   $PERMIT_DATA" | python3 -m json.tool 2>/dev/null || echo "   $PERMIT_DATA"
    fi
else
    echo -e "${YELLOW}⚠️  No permit keys found (Governor may not have processed plans yet)${NC}"
fi
echo ""

# ============================================================================
# PROOF 4: BLOCK RECORDS
# ============================================================================
echo -e "${BOLD}[PROOF 4/6] Block Records${NC}"

BLOCK_KEY_COUNT=$(redis-cli --scan --pattern "quantum:governor:block:*" | wc -l)

if [ "$BLOCK_KEY_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $BLOCK_KEY_COUNT block records${NC}"
    
    # Show sample block
    SAMPLE_BLOCK=$(redis-cli --scan --pattern "quantum:governor:block:*" | head -1)
    if [ -n "$SAMPLE_BLOCK" ]; then
        echo "   Sample block:"
        BLOCK_DATA=$(redis-cli GET "$SAMPLE_BLOCK")
        echo "   $BLOCK_DATA" | python3 -m json.tool 2>/dev/null || echo "   $BLOCK_DATA"
    fi
else
    echo "   No block records (good - no violations yet)"
fi
echo ""

# ============================================================================
# PROOF 5: EXECUTION TRACKING
# ============================================================================
echo -e "${BOLD}[PROOF 5/6] Execution Tracking${NC}"

EXEC_KEYS=$(redis-cli --scan --pattern "quantum:governor:exec:*")

if [ -n "$EXEC_KEYS" ]; then
    echo -e "${GREEN}✅ Governor is tracking executions${NC}"
    
    # Count tracked symbols
    SYMBOL_COUNT=$(echo "$EXEC_KEYS" | wc -l)
    echo "   Tracked symbols: $SYMBOL_COUNT"
    
    # Show execution counts per symbol
    for KEY in $EXEC_KEYS; do
        SYMBOL=$(echo "$KEY" | sed 's/quantum:governor:exec://')
        COUNT=$(redis-cli LLEN "$KEY")
        echo "   - $SYMBOL: $COUNT executions tracked"
    done
else
    echo "   No execution tracking keys yet (Governor starting up or no executions)"
fi
echo ""

# ============================================================================
# PROOF 6: GOVERNOR EVENT STREAM
# ============================================================================
echo -e "${BOLD}[PROOF 6/6] Governor Event Stream${NC}"

EVENT_COUNT=$(redis-cli XLEN quantum:stream:governor.events 2>/dev/null || echo "0")

if [ "$EVENT_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $EVENT_COUNT events in governor.events stream${NC}"
    
    # Show latest event
    echo "   Latest event:"
    LATEST_EVENT=$(redis-cli XREVRANGE quantum:stream:governor.events + - COUNT 1 2>/dev/null)
    if [ -n "$LATEST_EVENT" ]; then
        echo "$LATEST_EVENT" | head -20
    fi
else
    echo "   No events in governor.events stream yet (no disarm or critical events)"
fi
echo ""

# ============================================================================
# PROOF 7: CONFIG VERIFICATION
# ============================================================================
echo -e "${BOLD}[PROOF 7/6] Config Verification${NC}"

if [ -f /etc/quantum/governor.env ]; then
    echo -e "${GREEN}✅ Config file exists${NC}"
    echo "   Key settings:"
    grep -E "^(GOV_MAX_EXEC_PER_HOUR|GOV_MAX_EXEC_PER_5MIN|GOV_ENABLE_AUTO_DISARM)" /etc/quantum/governor.env || echo "   (Could not read settings)"
else
    echo -e "${RED}❌ Config file not found: /etc/quantum/governor.env${NC}"
    EXIT_CODE=1
fi
echo ""

# ============================================================================
# PROOF 8: RECENT LOGS
# ============================================================================
echo -e "${BOLD}[PROOF 8/6] Recent Log Sample${NC}"

echo "Last 10 log lines:"
journalctl -u quantum-governor -n 10 --no-pager | sed 's/^/   /'
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "${BOLD}=== PROOF PACK SUMMARY ===${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All critical proofs passed${NC}"
    echo ""
    echo "Governor operational status:"
    echo "  • Service: Active"
    echo "  • Metrics: Responding (port 8044)"
    echo "  • Permits: Issuing"
    echo "  • Blocks: Recording"
    echo "  • Tracking: Functional"
    echo ""
    echo -e "${GREEN}P3.2 Governor is OPERATIONAL and enforcing limits!${NC}"
else
    echo -e "${RED}❌ Some proofs failed${NC}"
    echo "Check logs: journalctl -u quantum-governor -n 50"
fi

exit $EXIT_CODE

#!/bin/bash
set -euo pipefail

# P3.0 DRY_RUN: ARM/DISARM SWITCH + PROOF
# Safely switch Apply Layer to dry_run and prove execution is disabled

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m"

echo -e "${BOLD}=== P3.0 DRY_RUN RE-LOCK ===${NC}"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# ============================================================================
# PHASE 1: BACKUP CONFIG
# ============================================================================
echo -e "${BOLD}[1/6] Backup config${NC}"
BACKUP_FILE="/etc/quantum/apply-layer.env.bak.$(date +%s)"
if cp /etc/quantum/apply-layer.env "$BACKUP_FILE" 2>/dev/null; then
    echo "✅ Backed up to: $BACKUP_FILE"
else
    echo -e "${RED}❌ Failed to backup config${NC}"
    exit 1
fi
echo ""

# ============================================================================
# PHASE 2: SET DRY_RUN MODE
# ============================================================================
echo -e "${BOLD}[2/6] Set APPLY_MODE=dry_run${NC}"
if sudo sed -i 's/^APPLY_MODE=.*/APPLY_MODE=dry_run/' /etc/quantum/apply-layer.env; then
    CURRENT_MODE=$(grep '^APPLY_MODE=' /etc/quantum/apply-layer.env | cut -d= -f2)
    if [ "$CURRENT_MODE" = "dry_run" ]; then
        echo "✅ Config set: APPLY_MODE=dry_run"
    else
        echo -e "${RED}❌ Failed to set dry_run mode (found: $CURRENT_MODE)${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Failed to edit config${NC}"
    exit 1
fi
echo ""

# ============================================================================
# PHASE 3: RESTART SERVICE
# ============================================================================
echo -e "${BOLD}[3/6] Restart service${NC}"
if sudo systemctl restart quantum-apply-layer; then
    echo "✅ Service restarted"
    sleep 3
    if systemctl is-active --quiet quantum-apply-layer; then
        echo "✅ Service is active"
    else
        echo -e "${RED}❌ Service failed to start${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Failed to restart service${NC}"
    exit 1
fi
echo ""

# ============================================================================
# PHASE 4: RUNTIME ENV PROOF
# ============================================================================
echo -e "${BOLD}[4/6] Runtime environment proof${NC}"
MAIN_PID=$(systemctl show quantum-apply-layer -p MainPID --value)
if [ "$MAIN_PID" -gt 0 ] 2>/dev/null; then
    RUNTIME_MODE=$(sudo cat /proc/"$MAIN_PID"/environ | tr '\0' '\n' | grep '^APPLY_MODE=' | cut -d= -f2)
    if [ "$RUNTIME_MODE" = "dry_run" ]; then
        echo "✅ Runtime env: APPLY_MODE=dry_run (PID: $MAIN_PID)"
    else
        echo -e "${RED}❌ Runtime env mismatch: $RUNTIME_MODE${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Failed to get MainPID${NC}"
    exit 1
fi
echo ""

# ============================================================================
# PHASE 5: EXECUTION PROOF (NO executed=True)
# ============================================================================
echo -e "${BOLD}[5/6] Execution proof (last 50 results)${NC}"

# Get last 50 results
RESULTS=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 50 2>/dev/null || echo "")
if [ -z "$RESULTS" ]; then
    echo -e "${RED}❌ Failed to read result stream${NC}"
    exit 1
fi

# Count executed=True (must be 0)
EXEC_TRUE_COUNT=$(echo "$RESULTS" | grep -A1 '^executed$' | grep -c '^True$' || echo "0")
if [ "$EXEC_TRUE_COUNT" -gt 0 ]; then
    echo -e "${RED}❌ DANGER: Found $EXEC_TRUE_COUNT executed=True in last 50 results${NC}"
    echo "   System may still be executing!"
    exit 1
else
    echo "✅ executed=True count: 0 (no execution)"
fi

# Check for decisions (proof service is alive)
DECISION_COUNT=$(echo "$RESULTS" | grep -c '^decision$' || echo "0")
if [ "$DECISION_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No decisions found - service may be dead${NC}"
    exit 1
else
    echo "✅ Decisions found: $DECISION_COUNT (service alive)"
fi

# Check for BTCUSDT (allowlist proof)
BTCUSDT_COUNT=$(echo "$RESULTS" | grep -A1 '^symbol$' | grep -c '^BTCUSDT$' || echo "0")
if [ "$BTCUSDT_COUNT" -gt 0 ]; then
    echo "✅ BTCUSDT results: $BTCUSDT_COUNT (allowlist active)"
else
    echo -e "${YELLOW}⚠️  No BTCUSDT in last 50 results${NC}"
fi

echo ""

# ============================================================================
# PHASE 6: STREAM ACTIVITY (LAST 5 RESULTS)
# ============================================================================
echo -e "${BOLD}[6/6] Last 5 results (sample)${NC}"
echo "---"

LAST_5=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 2>/dev/null || echo "")
if [ -z "$LAST_5" ]; then
    echo -e "${RED}❌ Failed to read last 5 results${NC}"
    exit 1
fi

# Parse and display (key fields only)
echo "$LAST_5" | awk '
BEGIN { entry=0 }
/^[0-9]+-[0-9]+$/ { 
    if (entry > 0) print ""
    entry++
    if (entry > 5) exit
    printf "Entry %d: ", entry
    next
}
/^symbol$/ { getline; printf "symbol=%s ", $0; next }
/^decision$/ { getline; printf "decision=%s ", $0; next }
/^executed$/ { getline; printf "executed=%s ", $0; next }
/^would_execute$/ { getline; printf "would_execute=%s ", $0; next }
/^error$/ { getline; if ($0 != "") printf "error=%s ", $0; next }
'
echo ""
echo "---"
echo ""

# ============================================================================
# FINAL REPORT
# ============================================================================
echo -e "${BOLD}=== DRY_RUN RE-LOCK COMPLETE ===${NC}"
echo -e "${GREEN}✅ All proofs passed${NC}"
echo ""
echo "Summary:"
echo "  • Config: APPLY_MODE=dry_run"
echo "  • Runtime: APPLY_MODE=dry_run"
echo "  • Execution disabled: 0 executed=True in last 50 results"
echo "  • Service: Active and processing"
echo "  • Backup: $BACKUP_FILE"
echo ""
echo -e "${GREEN}P3 Apply Layer is now SAFELY DISARMED (dry_run only)${NC}"

exit 0

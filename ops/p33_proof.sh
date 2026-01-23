#!/usr/bin/env bash
# P3.3 Position State Brain - Proof Pack
#
# VERIFIES:
# - P3.3 service active and healthy
# - Exchange snapshots updating
# - Ledger tracking working
# - Permit issuance operational
# - Sanity checks enforced
# - Apply Layer integration working

set -euo pipefail

echo "=============================="
echo "P3.3 PROOF PACK"
echo "=============================="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Configuration
REDIS_CLI="redis-cli"
SERVICE_NAME="quantum-position-state-brain"
METRICS_PORT="8045"

# Proof 1: Service Status
echo "[PROOF 1/8] Service Status"
echo "---------------------------"
systemctl status "${SERVICE_NAME}" --no-pager | head -15
echo ""

# Proof 2: Metrics Endpoint
echo "[PROOF 2/8] Metrics Endpoint (port ${METRICS_PORT})"
echo "----------------------------------------------------"
METRICS=$(curl -s http://localhost:${METRICS_PORT}/metrics 2>&1 || echo "FAILED")
if echo "${METRICS}" | grep -q "p33_"; then
    echo "✓ P3.3 metrics responding"
    echo ""
    echo "Sample metrics:"
    echo "${METRICS}" | grep "^p33_" | head -10
else
    echo "❌ FAILED: No P3.3 metrics found"
    echo "${METRICS}"
fi
echo ""

# Proof 3: Exchange Snapshots
echo "[PROOF 3/8] Exchange Position Snapshots"
echo "----------------------------------------"
SNAPSHOT_KEYS=$($REDIS_CLI KEYS "quantum:position:snapshot:*" 2>/dev/null || echo "")
if [ -z "${SNAPSHOT_KEYS}" ]; then
    echo "⚠ No snapshot keys found (may need time to populate)"
else
    echo "Found snapshots:"
    for key in ${SNAPSHOT_KEYS}; do
        echo ""
        echo "Key: ${key}"
        $REDIS_CLI HGETALL "${key}" | head -20
        
        # Check freshness
        TS=$($REDIS_CLI HGET "${key}" ts_epoch 2>/dev/null || echo "0")
        NOW=$(date +%s)
        AGE=$((NOW - TS))
        echo "Age: ${AGE}s"
        
        if [ "${AGE}" -lt 15 ]; then
            echo "✓ Fresh (< 15s)"
        else
            echo "⚠ Stale (> 15s)"
        fi
    done
fi
echo ""

# Proof 4: Internal Ledger
echo "[PROOF 4/8] Internal Position Ledger"
echo "-------------------------------------"
LEDGER_KEYS=$($REDIS_CLI KEYS "quantum:position:ledger:*" 2>/dev/null || echo "")
if [ -z "${LEDGER_KEYS}" ]; then
    echo "⚠ No ledger keys found (may need execution to populate)"
else
    echo "Found ledgers:"
    for key in ${LEDGER_KEYS}; do
        echo ""
        echo "Key: ${key}"
        $REDIS_CLI HGETALL "${key}" | head -20
    done
fi
echo ""

# Proof 5: Permit Keys
echo "[PROOF 5/8] P3.3 Permit Keys"
echo "----------------------------"
PERMIT_KEYS=$($REDIS_CLI KEYS "quantum:permit:p33:*" 2>/dev/null || echo "")
if [ -z "${PERMIT_KEYS}" ]; then
    echo "⚠ No P3.3 permit keys found (normal if no active plans)"
else
    echo "Found permits:"
    for key in ${PERMIT_KEYS}; do
        echo ""
        echo "Key: ${key}"
        PERMIT_DATA=$($REDIS_CLI GET "${key}" 2>/dev/null || echo "{}")
        echo "${PERMIT_DATA}" | python3 -m json.tool 2>/dev/null || echo "${PERMIT_DATA}"
    done
fi
echo ""

# Proof 6: Recent Logs (sanity checks)
echo "[PROOF 6/8] Recent P3.3 Logs (sanity checks)"
echo "---------------------------------------------"
journalctl -u "${SERVICE_NAME}" --since "5 minutes ago" --no-pager | grep -E "(PERMIT|DENY|BLOCK|safe_close_qty)" | tail -20 || echo "No recent sanity check logs"
echo ""

# Proof 7: Apply Layer Integration
echo "[PROOF 7/8] Apply Layer P3.3 Integration"
echo "-----------------------------------------"
echo "Checking Apply Layer logs for P3.3 permit checks..."
journalctl -u quantum-apply-layer --since "5 minutes ago" --no-pager | grep -E "(P3.3|p33_permit)" | tail -10 || echo "No recent P3.3 logs in Apply Layer"
echo ""

# Proof 8: Health Summary
echo "[PROOF 8/8] Health Summary"
echo "--------------------------"
echo "Services:"
echo "  - P3.3 Brain: $(systemctl is-active ${SERVICE_NAME})"
echo "  - Apply Layer: $(systemctl is-active quantum-apply-layer)"
echo "  - Governor: $(systemctl is-active quantum-governor)"
echo ""

echo "Redis Keys:"
SNAPSHOT_COUNT=$($REDIS_CLI KEYS "quantum:position:snapshot:*" 2>/dev/null | wc -l || echo "0")
LEDGER_COUNT=$($REDIS_CLI KEYS "quantum:position:ledger:*" 2>/dev/null | wc -l || echo "0")
PERMIT_COUNT=$($REDIS_CLI KEYS "quantum:permit:p33:*" 2>/dev/null | wc -l || echo "0")
echo "  - Snapshots: ${SNAPSHOT_COUNT}"
echo "  - Ledgers: ${LEDGER_COUNT}"
echo "  - P3.3 Permits: ${PERMIT_COUNT}"
echo ""

echo "Metrics:"
SNAPSHOT_TOTAL=$(curl -s http://localhost:${METRICS_PORT}/metrics 2>/dev/null | grep "^p33_snapshot_total" | tail -1 || echo "N/A")
PERMIT_TOTAL=$(curl -s http://localhost:${METRICS_PORT}/metrics 2>/dev/null | grep "^p33_permit_total" | tail -1 || echo "N/A")
echo "  - Snapshot updates: ${SNAPSHOT_TOTAL}"
echo "  - Permits issued: ${PERMIT_TOTAL}"
echo ""

echo "=============================="
echo "PROOF PACK COMPLETE"
echo "=============================="
echo ""
echo "INSTRUCTIONS:"
echo "- Review snapshots are fresh (< 15s)"
echo "- Check permits being issued/denied"
echo "- Verify Apply Layer checking P3.3 permits"
echo "- Monitor logs for sanity check outcomes"
echo ""
echo "Expected behavior:"
echo "- P3.3 updates snapshots every 5s"
echo "- P3.3 evaluates EXECUTE plans and issues permits"
echo "- Apply Layer requires BOTH Governor + P3.3 permits"
echo "- Fail-closed: missing permit → execution BLOCKED"

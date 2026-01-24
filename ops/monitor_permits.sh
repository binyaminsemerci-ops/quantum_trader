#!/bin/bash
# Monitor P3.2 Governor and P3.3 Permit Metrics

set -e

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
GOV_METRICS_PORT="${GOV_METRICS_PORT:-8044}"
P33_METRICS_PORT="${P33_METRICS_PORT:-8045}"

echo "==========================================="
echo "Permit System Monitoring"
echo "==========================================="
echo ""

# Service status
echo "1. Service Status"
echo "------------------"
systemctl is-active --quiet quantum-governor && echo "✓ Governor: RUNNING" || echo "✗ Governor: STOPPED"
systemctl is-active --quiet quantum-position-state-brain && echo "✓ P3.3: RUNNING" || echo "✗ P3.3: STOPPED"
systemctl is-active --quiet quantum-apply-layer && echo "✓ Apply Layer: RUNNING" || echo "✗ Apply Layer: STOPPED"
echo ""

# Consumer groups
echo "2. Consumer Groups"
echo "------------------"
echo "Governor consumer group:"
redis-cli -h $REDIS_HOST -p $REDIS_PORT XINFO GROUPS quantum:stream:apply.plan 2>/dev/null | grep -A5 "name.*governor" || echo "  Not found"
echo ""
echo "P3.3 consumer group:"
redis-cli -h $REDIS_HOST -p $REDIS_PORT XINFO GROUPS quantum:stream:apply.plan 2>/dev/null | grep -A5 "name.*p33" || echo "  Not found"
echo ""

# Governor metrics
echo "3. Governor Metrics (P3.2)"
echo "--------------------------"
if curl -s http://localhost:$GOV_METRICS_PORT/metrics > /dev/null 2>&1; then
    echo "Governor permits issued:"
    curl -s http://localhost:$GOV_METRICS_PORT/metrics | grep "^quantum_govern_allow_total" | head -5
    echo ""
    echo "Governor permits blocked:"
    curl -s http://localhost:$GOV_METRICS_PORT/metrics | grep "^quantum_govern_block_total" | head -5
    echo ""
    echo "Execution counts:"
    curl -s http://localhost:$GOV_METRICS_PORT/metrics | grep "^quantum_govern_exec_count" | head -10
else
    echo "  ✗ Metrics not available on port $GOV_METRICS_PORT"
fi
echo ""

# P3.3 metrics
echo "4. P3.3 Metrics"
echo "---------------"
if curl -s http://localhost:$P33_METRICS_PORT/metrics > /dev/null 2>&1; then
    echo "P3.3 permits issued:"
    curl -s http://localhost:$P33_METRICS_PORT/metrics | grep "^p33_permit_allow_total" | head -5
    echo ""
    echo "P3.3 permits denied:"
    curl -s http://localhost:$P33_METRICS_PORT/metrics | grep "^p33_permit_deny_total" | head -5
    echo ""
    echo "P3.3 position tracking:"
    curl -s http://localhost:$P33_METRICS_PORT/metrics | grep "^p33_.*_amt" | head -10
else
    echo "  ✗ Metrics not available on port $P33_METRICS_PORT"
fi
echo ""

# Permit ratios
echo "5. Permit Decision Ratios"
echo "-------------------------"

# Governor ratio
GOV_ALLOW=$(curl -s http://localhost:$GOV_METRICS_PORT/metrics 2>/dev/null | grep "^quantum_govern_allow_total{" | awk '{sum+=$2} END {print sum+0}')
GOV_BLOCK=$(curl -s http://localhost:$GOV_METRICS_PORT/metrics 2>/dev/null | grep "^quantum_govern_block_total{" | awk '{sum+=$2} END {print sum+0}')
GOV_TOTAL=$((GOV_ALLOW + GOV_BLOCK))

if [ "$GOV_TOTAL" -gt 0 ]; then
    GOV_ALLOW_PCT=$(awk "BEGIN {printf \"%.1f\", ($GOV_ALLOW / $GOV_TOTAL) * 100}")
    echo "Governor: $GOV_ALLOW allowed / $GOV_BLOCK blocked (${GOV_ALLOW_PCT}% approval rate)"
else
    echo "Governor: No decisions yet"
fi

# P3.3 ratio
P33_ALLOW=$(curl -s http://localhost:$P33_METRICS_PORT/metrics 2>/dev/null | grep "^p33_permit_allow_total{" | awk '{sum+=$2} END {print sum+0}')
P33_DENY=$(curl -s http://localhost:$P33_METRICS_PORT/metrics 2>/dev/null | grep "^p33_permit_deny_total{" | awk '{sum+=$2} END {print sum+0}')
P33_TOTAL=$((P33_ALLOW + P33_DENY))

if [ "$P33_TOTAL" -gt 0 ]; then
    P33_ALLOW_PCT=$(awk "BEGIN {printf \"%.1f\", ($P33_ALLOW / $P33_TOTAL) * 100}")
    echo "P3.3: $P33_ALLOW allowed / $P33_DENY denied (${P33_ALLOW_PCT}% approval rate)"
else
    echo "P3.3: No decisions yet"
fi
echo ""

# Recent logs
echo "6. Recent Permit Activity (last 2 minutes)"
echo "-------------------------------------------"
echo "Governor:"
journalctl -u quantum-governor --since "2 minutes ago" --no-pager -o cat 2>/dev/null | grep -E "auto-approving|ALLOW|DENY|permit issued|permit blocked" | tail -5 || echo "  No recent activity"
echo ""
echo "P3.3:"
journalctl -u quantum-position-state-brain --since "2 minutes ago" --no-pager -o cat 2>/dev/null | grep -E "P3.3 ALLOW|P3.3 DENY" | tail -5 || echo "  No recent activity"
echo ""

# Apply Layer execution status
echo "7. Recent Executions (last 2 minutes)"
echo "--------------------------------------"
journalctl -u quantum-apply-layer --since "2 minutes ago" --no-pager -o cat 2>/dev/null | grep -E "executed=True|executed=False|permit_timeout|permits ready" | tail -10 || echo "  No recent execution attempts"
echo ""

# Ledger reconciliation check
echo "8. Ledger Reconciliation Status"
echo "--------------------------------"
for SYMBOL in BTCUSDT ETHUSDT; do
    SNAPSHOT_AMT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT HGET quantum:position:snapshot:$SYMBOL position_amt 2>/dev/null || echo "N/A")
    LEDGER_AMT=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT HGET quantum:position:ledger:$SYMBOL last_known_amt 2>/dev/null || echo "N/A")
    
    if [ "$SNAPSHOT_AMT" != "N/A" ] && [ "$LEDGER_AMT" != "N/A" ]; then
        DIFF=$(awk "BEGIN {printf \"%.4f\", $SNAPSHOT_AMT - $LEDGER_AMT}")
        if [ "${DIFF#-}" != "0.0000" ]; then
            echo "⚠ $SYMBOL: snapshot=$SNAPSHOT_AMT, ledger=$LEDGER_AMT (diff=$DIFF)"
        else
            echo "✓ $SYMBOL: In sync (amt=$SNAPSHOT_AMT)"
        fi
    else
        echo "  $SYMBOL: snapshot=$SNAPSHOT_AMT, ledger=$LEDGER_AMT"
    fi
done
echo ""

echo "==========================================="
echo "Monitoring Complete"
echo "==========================================="

#!/bin/bash
# Quantum Trader - Pipeline Proof Pack
# Captures complete pipeline state for before/after deployment comparisons
#
# Usage:
#   ./proof_pipeline.sh > proof_$(date +%Y%m%d_%H%M%S).txt
#   ./proof_pipeline.sh --before  # Save as proof_before.txt
#   ./proof_pipeline.sh --after   # Save as proof_after.txt

set -euo pipefail

TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
MODE="${1:-}"

if [ "$MODE" == "--before" ]; then
    OUTPUT="/tmp/proof_before.txt"
elif [ "$MODE" == "--after" ]; then
    OUTPUT="/tmp/proof_after.txt"
else
    OUTPUT="/dev/stdout"
fi

{
    echo "=========================================="
    echo "QUANTUM TRADER - PIPELINE PROOF PACK"
    echo "=========================================="
    echo "Timestamp: $TIMESTAMP"
    echo ""

    # ============================================================================
    # EXECUTION.RESULT STREAM
    # ============================================================================
    echo "=== execution.result Stream Status ==="
    redis-cli XINFO STREAM quantum:stream:execution.result | grep -E "length|last-generated-id|entries-added|radix-tree-keys" | paste - - | column -t
    echo ""
    
    RESULT_LASTID=$(redis-cli XINFO STREAM quantum:stream:execution.result | awk '$0=="last-generated-id"{getline; print}')
    RESULT_LEN=$(redis-cli XLEN quantum:stream:execution.result)
    RESULT_ENTRIES=$(redis-cli XINFO STREAM quantum:stream:execution.result | awk '$0=="entries-added"{getline; print}')
    
    echo "Summary:"
    echo "  Last ID: $RESULT_LASTID"
    echo "  Length: $RESULT_LEN"
    echo "  Entries Added: $RESULT_ENTRIES"
    echo ""

    # ============================================================================
    # TRADE.INTENT CONSUMER GROUPS
    # ============================================================================
    echo "=== trade.intent Consumer Groups ==="
    redis-cli XINFO GROUPS quantum:stream:trade.intent | grep -E "name|pending|lag|entries-read" | paste - - | column -t
    echo ""
    
    # Get specific group stats
    INTENT_PENDING=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="pending"{getline; print; exit}')
    INTENT_LAG=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="lag"{getline; print; exit}')
    INTENT_ENTRIES=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="entries-read"{getline; print; exit}')
    
    echo "Execution Consumer Group:"
    echo "  Pending: $INTENT_PENDING"
    echo "  Lag: $INTENT_LAG"
    echo "  Entries Read: $INTENT_ENTRIES"
    echo ""
    
    # Calculate throughput if we have lag
    if [ "$INTENT_LAG" -gt 0 ]; then
        echo "⚠️  WARNING: Consumer lag detected: $INTENT_LAG messages behind"
    else
        echo "✅ Consumer group caught up (lag=0)"
    fi
    echo ""

    # ============================================================================
    # RATE LIMITER STATUS (P0.D.5)
    # ============================================================================
    echo "=== Rate Limiter Status ==="
    
    # Check for quantum:rate:* keys
    RATE_KEYS=$(redis-cli KEYS "quantum:rate:*" 2>/dev/null || echo "")
    
    if [ -z "$RATE_KEYS" ]; then
        echo "✅ No active rate limiters"
    else
        echo "Active limiters:"
        for key in $RATE_KEYS; do
            COUNT=$(redis-cli GET "$key" 2>/dev/null || echo "0")
            TTL=$(redis-cli TTL "$key" 2>/dev/null || echo "-1")
            if [ "$TTL" -gt 0 ]; then
                echo "  $key: $COUNT requests | Resets in ${TTL}s"
            else
                echo "  $key: $COUNT requests | No TTL"
            fi
        done
    fi
    
    # Calculate next reset ETA for quantum:rate:execution:global
    GLOBAL_RATE_KEY="quantum:rate:execution:global"
    GLOBAL_TTL=$(redis-cli TTL "$GLOBAL_RATE_KEY" 2>/dev/null || echo "-1")
    if [ "$GLOBAL_TTL" -gt 0 ]; then
        RESET_TIME=$(date -d "+${GLOBAL_TTL} seconds" +"%H:%M:%S" 2>/dev/null || echo "N/A")
        echo "Next global reset: $RESET_TIME (in ${GLOBAL_TTL}s)"
    fi
    echo ""

    # ============================================================================
    # SYSTEMD SERVICE STATUS
    # ============================================================================
    echo "=== Systemd Service Status ==="
    
    # Execution service
    if systemctl is-active --quiet quantum-execution; then
        echo "✅ quantum-execution.service: ACTIVE"
        EXEC_PID=$(systemctl show quantum-execution -p MainPID --value)
        EXEC_UPTIME=$(systemctl show quantum-execution -p ActiveEnterTimestamp --value)
        echo "   PID: $EXEC_PID"
        echo "   Started: $EXEC_UPTIME"
    else
        echo "❌ quantum-execution.service: INACTIVE"
    fi
    echo ""
    
    # AI Engine service
    if systemctl is-active --quiet quantum-ai-engine; then
        echo "✅ quantum-ai-engine.service: ACTIVE"
        AI_PID=$(systemctl show quantum-ai-engine -p MainPID --value)
        AI_UPTIME=$(systemctl show quantum-ai-engine -p ActiveEnterTimestamp --value)
        echo "   PID: $AI_PID"
        echo "   Started: $AI_UPTIME"
    else
        echo "❌ quantum-ai-engine.service: INACTIVE"
    fi
    echo ""

    # ============================================================================
    # RECENT EXECUTION LOGS
    # ============================================================================
    echo "=== Recent Execution Logs (last 50 lines, filtered) ==="
    tail -50 /var/log/quantum/execution.log | grep -vE "P0\.D\.4d.*stream=" | tail -30 || echo "(no recent logs)"
    echo ""

    # ============================================================================
    # ERROR SUMMARY
    # ============================================================================
    echo "=== Error Summary (last 100 lines) ==="
    ERROR_COUNT=$(tail -100 /var/log/quantum/execution.log | grep -ci "error\|failed\|exception" || echo "0")
    WARNING_COUNT=$(tail -100 /var/log/quantum/execution.log | grep -ci "warning" || echo "0")
    
    echo "Errors: $ERROR_COUNT"
    echo "Warnings: $WARNING_COUNT"
    
    if [ "$ERROR_COUNT" -gt 5 ]; then
        echo ""
        echo "Recent errors:"
        tail -100 /var/log/quantum/execution.log | grep -i "error\|failed\|exception" | tail -5 || true
    fi
    echo ""

    # ============================================================================
    # THROUGHPUT ESTIMATE
    # ============================================================================
    echo "=== Throughput Estimate (5-second sample) ==="
    
    ENTRIES_START=$INTENT_ENTRIES
    sleep 5
    ENTRIES_END=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="entries-read"{getline; print; exit}')
    ENTRIES_DELTA=$((ENTRIES_END - ENTRIES_START))
    MSGS_PER_MIN=$((ENTRIES_DELTA * 12))
    
    echo "Messages processed: $ENTRIES_DELTA (in 5 seconds)"
    echo "Estimated rate: $MSGS_PER_MIN messages/minute"
    echo ""

    # ============================================================================
    # HEALTH CHECK
    # ============================================================================
    echo "=== Health Check Summary ==="
    
    HEALTH_OK=true
    
    # Check execution service
    if ! systemctl is-active --quiet quantum-execution; then
        echo "❌ FAIL: quantum-execution.service not running"
        HEALTH_OK=false
    fi
    
    # Check lag
    if [ "$INTENT_LAG" -gt 100000 ]; then
        echo "⚠️  WARN: High consumer lag: $INTENT_LAG"
    fi
    
    # Check pending
    if [ "$INTENT_PENDING" -gt 50000 ]; then
        echo "⚠️  WARN: High pending messages: $INTENT_PENDING"
    fi
    
    # Check errors
    if [ "$ERROR_COUNT" -gt 10 ]; then
        echo "⚠️  WARN: High error rate: $ERROR_COUNT errors in last 100 lines"
    fi
    
    if [ "$HEALTH_OK" = true ]; then
        echo "✅ All health checks passed"
    fi
    echo ""
    
    echo "=========================================="
    echo "END OF PROOF PACK"
    echo "=========================================="

} | if [ "$OUTPUT" != "/dev/stdout" ]; then
    tee "$OUTPUT"
    echo ""
    echo "✅ Proof pack saved to: $OUTPUT"
else
    cat
fi

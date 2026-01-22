#!/bin/bash
# Quantum Trader - Backlog Monitor
# Real-time monitoring of consumer lag and throughput
#
# Usage:
#   ./monitor_backlog.sh                    # Watch mode (updates every 5s)
#   ./monitor_backlog.sh --once             # Single snapshot
#   ./monitor_backlog.sh --alert 500000     # Alert if lag > threshold

set -euo pipefail

MODE="${1:-watch}"
THRESHOLD="${2:-1000000}"
REFRESH_INTERVAL=5

print_status() {
    clear
    echo "=========================================="
    echo "QUANTUM TRADER - BACKLOG MONITOR"
    echo "=========================================="
    echo "Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo ""
    
    # Get trade.intent consumer group stats
    echo "=== trade.intent Consumer Group ==="
    
    GROUP_INFO=$(redis-cli XINFO GROUPS quantum:stream:trade.intent 2>/dev/null || echo "")
    
    if [ -z "$GROUP_INFO" ]; then
        echo "❌ ERROR: Cannot read consumer group info"
        return 1
    fi
    
    PENDING=$(echo "$GROUP_INFO" | awk '$0=="pending"{getline; print; exit}')
    LAG=$(echo "$GROUP_INFO" | awk '$0=="lag"{getline; print; exit}')
    ENTRIES_READ=$(echo "$GROUP_INFO" | awk '$0=="entries-read"{getline; print; exit}')
    LAST_DELIVERED=$(echo "$GROUP_INFO" | awk '$0=="last-delivered-id"{getline; print; exit}')
    
    echo "Pending:        $PENDING"
    echo "Lag:            $LAG"
    echo "Entries Read:   $ENTRIES_READ"
    echo "Last Delivered: $LAST_DELIVERED"
    echo ""
    
    # Status indicators
    if [ "$LAG" -gt "$THRESHOLD" ]; then
        echo "🔴 CRITICAL: Lag exceeds threshold ($LAG > $THRESHOLD)"
        ALERT=true
    elif [ "$LAG" -gt $((THRESHOLD / 2)) ]; then
        echo "🟡 WARNING: Lag approaching threshold ($LAG > $((THRESHOLD / 2)))"
        ALERT=false
    else
        echo "🟢 OK: Lag within acceptable range"
        ALERT=false
    fi
    echo ""
    
    # Calculate backlog percentage (if we know stream length)
    STREAM_LEN=$(redis-cli XLEN quantum:stream:trade.intent 2>/dev/null || echo "0")
    if [ "$STREAM_LEN" -gt 0 ]; then
        BACKLOG_PCT=$((LAG * 100 / STREAM_LEN))
        echo "Backlog: $BACKLOG_PCT% of stream"
        echo ""
    fi
    
    # Throughput estimate (5-second sample)
    echo "=== Throughput Estimate ==="
    if [ "$MODE" == "watch" ]; then
        ENTRIES_START=$ENTRIES_READ
        sleep 5
        ENTRIES_END=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="entries-read"{getline; print; exit}')
        ENTRIES_DELTA=$((ENTRIES_END - ENTRIES_START))
        MSGS_PER_SEC=$(echo "scale=2; $ENTRIES_DELTA / 5" | bc)
        MSGS_PER_MIN=$((ENTRIES_DELTA * 12))
        
        echo "Messages/sec:    $MSGS_PER_SEC"
        echo "Messages/min:    $MSGS_PER_MIN"
        echo "Processed:       $ENTRIES_DELTA (in 5s sample)"
        echo ""
        
        # ETA to clear backlog
        if [ "$LAG" -gt 0 ] && [ "$ENTRIES_DELTA" -gt 0 ]; then
            ETA_SECONDS=$((LAG / (ENTRIES_DELTA / 5)))
            ETA_MINUTES=$((ETA_SECONDS / 60))
            ETA_HOURS=$((ETA_MINUTES / 60))
            
            echo "=== Backlog Clearance ETA ==="
            if [ "$ETA_HOURS" -gt 24 ]; then
                echo "ETA: $((ETA_HOURS / 24)) days"
            elif [ "$ETA_HOURS" -gt 0 ]; then
                echo "ETA: $ETA_HOURS hours $((ETA_MINUTES % 60)) minutes"
            else
                echo "ETA: $ETA_MINUTES minutes"
            fi
            echo "(at current throughput rate)"
            echo ""
        fi
    fi
    
    # Recent execution activity
    echo "=== Recent Execution Activity ==="
    FILLED_COUNT=$(tail -50 /var/log/quantum/execution.log | grep -c "FILLED" 2>/dev/null || echo "0")
    RATE_LIMITED=$(tail -50 /var/log/quantum/execution.log | grep -c "RATE_LIMIT" 2>/dev/null || echo "0")
    
    echo "FILLED orders (last 50 lines):  $FILLED_COUNT"
    echo "Rate limited (last 50 lines):   $RATE_LIMITED"
    echo ""
    
    # System health
    echo "=== System Health ==="
    if systemctl is-active --quiet quantum-execution; then
        EXEC_PID=$(systemctl show quantum-execution -p MainPID --value)
        EXEC_MEM=$(ps -p "$EXEC_PID" -o rss= 2>/dev/null | awk '{print int($1/1024)"MB"}' || echo "N/A")
        echo "✅ quantum-execution.service: ACTIVE (PID: $EXEC_PID, MEM: $EXEC_MEM)"
    else
        echo "❌ quantum-execution.service: INACTIVE"
    fi
    
    if systemctl is-active --quiet quantum-ai-engine; then
        AI_PID=$(systemctl show quantum-ai-engine -p MainPID --value)
        AI_MEM=$(ps -p "$AI_PID" -o rss= 2>/dev/null | awk '{print int($1/1024)"MB"}' || echo "N/A")
        echo "✅ quantum-ai-engine.service: ACTIVE (PID: $AI_PID, MEM: $AI_MEM)"
    else
        echo "❌ quantum-ai-engine.service: INACTIVE"
    fi
    echo ""
    
    if [ "$MODE" == "watch" ]; then
        echo "Press Ctrl+C to exit | Refreshing in ${REFRESH_INTERVAL}s..."
    fi
    
    # Return alert status for --alert mode
    if [ "$MODE" == "alert" ] && [ "$ALERT" = true ]; then
        return 1
    fi
    
    return 0
}

# Main loop
case "$MODE" in
    --once)
        print_status
        ;;
    --alert)
        if ! print_status; then
            echo ""
            echo "🚨 ALERT: Sending notification..."
            # Add your alerting logic here (email, Slack, PagerDuty, etc.)
            exit 1
        fi
        ;;
    watch|*)
        while true; do
            print_status || true
            sleep "$REFRESH_INTERVAL"
        done
        ;;
esac

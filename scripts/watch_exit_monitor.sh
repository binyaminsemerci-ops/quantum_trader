#!/bin/bash
# Exit-Monitor TP/SL Trigger Monitoring Script
# Watches for first exit signal and captures full details

VPS="root@46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"
LOG_FILE="exit_monitor_watch_$(date +%Y%m%d_%H%M%S).log"

echo "🔍 Starting Exit-Monitor TP/SL Watch..."
echo "📝 Logging to: $LOG_FILE"
echo "⏰ Started: $(date)"
echo ""

# Function to check health
check_health() {
    ssh -i $SSH_KEY $VPS "curl -s http://localhost:8007/health | python3 -m json.tool"
}

# Function to watch logs
watch_logs() {
    ssh -i $SSH_KEY $VPS "tail -f /var/log/quantum/exit-monitor.log | grep --line-buffered -E 'TP_HIT|SL_HIT|EXIT|CLOSE|TRACKING'"
}

# Function to check positions
check_positions() {
    ssh -i $SSH_KEY $VPS "redis-cli XINFO GROUPS quantum:stream:trade.execution.res | grep -E 'entries-read|lag|pending'"
}

# Initial status
echo "=== INITIAL STATUS ==="
check_health
echo ""
echo "=== CONSUMER GROUP ==="
check_positions
echo ""
echo "=== NOW WATCHING LOGS (Ctrl+C to stop) ==="
echo ""

# Watch logs and save to file
watch_logs | tee -a $LOG_FILE

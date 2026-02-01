#!/bin/bash
# Extended Validation - Real-Time Monitoring Script
# Usage: ./validate_monitor.sh [vps_host]

VPS_HOST="${1:-46.224.116.254}"
SSH_KEY="$HOME/.ssh/hetzner_fresh"
METRICS_FILE="validation_metrics_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Extended Validation Monitor Starting"
echo "VPS: $VPS_HOST"
echo "Metrics: $METRICS_FILE"
echo ""

# Function: Capture metrics
capture_metrics() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Get metrics from VPS
    local metrics=$(ssh -i "$SSH_KEY" root@"$VPS_HOST" '
        POSITIONS=$(redis-cli --raw HLEN quantum:ledger:latest || echo 0)
        EXPOSURE=$(redis-cli --raw GET quantum:portfolio:exposure_pct || echo "N/A")
        INTENTS=$(redis-cli XLEN quantum:stream:trade.intent || echo 0)
        PLANS=$(redis-cli XLEN quantum:stream:apply.plan || echo 0)
        EQUITY=$(redis-cli --raw GET quantum:account:equity_usd || echo "N/A")
        NOTIONAL=$(redis-cli --raw GET quantum:portfolio:notional_usd || echo "N/A")
        
        echo "$POSITIONS|$EXPOSURE|$INTENTS|$PLANS|$EQUITY|$NOTIONAL"
    ')
    
    IFS='|' read -r positions exposure intents plans equity notional <<< "$metrics"
    
    # Format output
    echo "$timestamp | Pos: $positions | Exp: $exposure% | Intents: $intents | Plans: $plans | Eq: $equity | Not: $notional" | tee -a "$METRICS_FILE"
}

# Function: Check services
check_services() {
    echo ""
    echo "=== SERVICE STATUS ==="
    ssh -i "$SSH_KEY" root@"$VPS_HOST" 'systemctl status quantum-trading-bot quantum-intent-bridge quantum-ai-engine --no-pager | grep -E "Active:|●"'
}

# Function: Get recent logs
get_recent_logs() {
    echo ""
    echo "=== RECENT TRADING BOT ACTIVITY ==="
    ssh -i "$SSH_KEY" root@"$VPS_HOST" 'journalctl -u quantum-trading-bot -n 5 --no-pager | tail -5'
    
    echo ""
    echo "=== RECENT INTENT BRIDGE ACTIVITY ==="
    ssh -i "$SSH_KEY" root@"$VPS_HOST" 'journalctl -u quantum-intent-bridge -n 5 --no-pager | tail -5'
}

# Function: Check for errors
check_errors() {
    echo ""
    echo "=== ERROR CHECK ==="
    local errors=$(ssh -i "$SSH_KEY" root@"$VPS_HOST" 'journalctl -u quantum-intent-bridge --since "5 minutes ago" | grep -i "error\|exception" | wc -l')
    echo "Errors in last 5 min: $errors"
    if [ "$errors" -gt 0 ]; then
        echo "⚠️  Found errors:"
        ssh -i "$SSH_KEY" root@"$VPS_HOST" 'journalctl -u quantum-intent-bridge --since "5 minutes ago" | grep -i "error\|exception"'
    fi
}

# Main loop
echo "Starting real-time monitoring (Ctrl+C to stop)..."
echo ""

iteration=0
while true; do
    iteration=$((iteration + 1))
    
    clear
    echo "🚀 QUANTUM TRADER - LIVE VALIDATION"
    echo "Iteration: $iteration | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    echo ""
    
    # Capture and display metrics
    capture_metrics
    
    # Every 3 iterations, show full details
    if [ $((iteration % 3)) -eq 0 ]; then
        check_services
        get_recent_logs
        check_errors
    fi
    
    echo ""
    echo "Last update: $(date '+%H:%M:%S') | Next update in 30 seconds..."
    sleep 30
done

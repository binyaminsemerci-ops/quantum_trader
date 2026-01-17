#!/bin/bash
# UTF/CLM Health Check Script

HEALTH_FILE="/var/lib/quantum/utf/health.json"
LOG_FILE="/var/log/quantum/utf_clm_health.log"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "$(timestamp) | $1" | tee -a "$LOG_FILE"
}

# Check if services are active
check_services() {
    local status="OK"
    
    if ! systemctl is-active --quiet quantum-utf-publisher.service; then
        log "FAIL: UTF Publisher service is not active"
        status="FAIL"
    fi
    
    if ! systemctl is-active --quiet quantum-clm-minimal.service; then
        log "FAIL: CLM Minimal service is not active"
        status="FAIL"
    fi
    
    if [ "$status" = "OK" ]; then
        log "OK: Both services are active"
    fi
    
    echo "$status"
}

# Check stream growth
check_stream_growth() {
    local current_len=$(redis-cli XLEN quantum:stream:utf 2>/dev/null || echo "0")
    local previous_len=0
    local status="OK"
    
    # Load previous length from health file
    if [ -f "$HEALTH_FILE" ]; then
        previous_len=$(jq -r '.utf_stream_len // 0' "$HEALTH_FILE" 2>/dev/null || echo "0")
        previous_timestamp=$(jq -r '.timestamp // 0' "$HEALTH_FILE" 2>/dev/null || echo "0")
    fi
    
    # Calculate growth
    local growth=$((current_len - previous_len))
    
    if [ "$growth" -lt 10 ] && [ "$previous_len" -gt 0 ]; then
        log "WARN: UTF stream growth is low: $growth events since last check"
        log "      Current: $current_len, Previous: $previous_len"
        status="WARN"
    else
        log "OK: UTF stream growing: +$growth events (total: $current_len)"
    fi
    
    # Save current state
    mkdir -p "$(dirname "$HEALTH_FILE")"
    cat > "$HEALTH_FILE" <<EOF
{
  "utf_stream_len": $current_len,
  "timestamp": $(date +%s),
  "growth": $growth
}
EOF
    
    echo "$status"
}

# Check CLM processing
check_clm_processing() {
    local error_count=$(redis-cli KEYS "quantum:clm:errors:*" 2>/dev/null | wc -l || echo "0")
    local counter_count=$(redis-cli KEYS "quantum:clm:count:*" 2>/dev/null | wc -l || echo "0")
    local status="OK"
    
    if [ "$counter_count" -eq 0 ]; then
        log "FAIL: CLM has not created any counters"
        status="FAIL"
    else
        log "OK: CLM active with $counter_count hourly counters, $error_count error trackers"
    fi
    
    echo "$status"
}

# Main health check
main() {
    log "=== UTF/CLM Health Check ==="
    
    service_status=$(check_services)
    growth_status=$(check_stream_growth)
    clm_status=$(check_clm_processing)
    
    # Extract just the status word (last line of each function)
    service_status=$(echo "$service_status" | tail -1)
    growth_status=$(echo "$growth_status" | tail -1)
    clm_status=$(echo "$clm_status" | tail -1)
    
    log "Status: services=$service_status, growth=$growth_status, clm=$clm_status"
    
    if [ "$service_status" = "OK" ] && [ "$growth_status" != "FAIL" ] && [ "$clm_status" = "OK" ]; then
        log "=== OVERALL: HEALTHY ==="
        exit 0
    else
        log "=== OVERALL: DEGRADED ==="
        exit 1
    fi
}

main

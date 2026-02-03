#!/usr/bin/env bash
#
# Exit Owner Watch - Monitor for unauthorized exit attempts
# =========================================================
#
# Checks logs for DENY_NOT_EXIT_OWNER events and alerts if found.
# Runs every 5 minutes via systemd timer.
#
# Usage:
#   bash scripts/exit_owner_watch.sh

set -euo pipefail

# Logging
LOG_PREFIX="[EXIT-OWNER-WATCH]"
log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} INFO: $1"
}

log_alert() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} 🚨 ALERT: $1" >&2
}

log_ok() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} ✅ OK: $1"
}

# Configuration
CHECK_WINDOW="5 minutes ago"
APPLY_LAYER_SERVICE="quantum-apply-layer"

# Check if service exists
if ! systemctl list-units --type=service 2>/dev/null | grep -q "$APPLY_LAYER_SERVICE"; then
    log_info "Service $APPLY_LAYER_SERVICE not found, skipping check"
    exit 0
fi

# Check for DENY_NOT_EXIT_OWNER events
DENY_COUNT=$(journalctl -u "$APPLY_LAYER_SERVICE" --since "$CHECK_WINDOW" --no-pager 2>/dev/null | grep -c "DENY_NOT_EXIT_OWNER" || echo "0")

# Check for ALLOW_EXIT_OWNER events (normal operation)
ALLOW_COUNT=$(journalctl -u "$APPLY_LAYER_SERVICE" --since "$CHECK_WINDOW" --no-pager 2>/dev/null | grep -c "ALLOW_EXIT_OWNER" || echo "0")

if [ "$DENY_COUNT" -gt 0 ]; then
    log_alert "Detected $DENY_COUNT unauthorized exit attempts in last 5 minutes"
    log_alert "Action: Review apply_layer logs for DENY_NOT_EXIT_OWNER"
    
    # Extract sample events
    echo ""
    echo "Sample unauthorized attempts:"
    journalctl -u "$APPLY_LAYER_SERVICE" --since "$CHECK_WINDOW" --no-pager 2>/dev/null | grep "DENY_NOT_EXIT_OWNER" | head -3
    echo ""
    
    # Write to Redis alert stream (best-effort)
    if command -v redis-cli &> /dev/null; then
        redis-cli XADD "quantum:stream:alerts" "*" \
            alert_type "EXIT_OWNER_VIOLATION" \
            deny_count "$DENY_COUNT" \
            window "5min" \
            timestamp "$(date +%s)" 2>/dev/null || true
    fi
    
    exit 1
else
    log_ok "No unauthorized exit attempts (checked last 5 minutes)"
    if [ "$ALLOW_COUNT" -gt 0 ]; then
        log_info "Authorized exits: $ALLOW_COUNT (exitbrain_v3_5)"
    fi
    exit 0
fi

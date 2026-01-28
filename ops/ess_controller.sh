#!/usr/bin/env bash
#
# ESS (Emergency Stop System) Controller
# OS-level emergency stop independent of Python/Redis
#
# Usage:
#   ess_controller.sh activate   - Activate ESS (stop trading)
#   ess_controller.sh deactivate - Deactivate ESS (restart services)
#   ess_controller.sh status     - Check ESS status
#

set -euo pipefail

# Config
ESS_FLAG_FILE="/var/run/quantum/ESS_ON"
ESS_FLAG_DIR="/var/run/quantum"
ESS_AUDIT_LOG="quantum-ess"

# Trading-critical services (MUST be stopped)
CRITICAL_SERVICES=(
    "quantum-ai-engine"
    "quantum-execution"
    "quantum-apply-layer"
    "quantum-governor"
)

# Monitoring services (can stay running)
MONITOR_SERVICES=(
    "quantum-rl-monitor"
    "grafana-server"
    "prometheus"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Ensure flag directory exists
mkdir -p "$ESS_FLAG_DIR"

log_audit() {
    local msg="$1"
    logger -t "$ESS_AUDIT_LOG" -p user.crit "$msg"
    echo -e "${BLUE}[ESS AUDIT]${NC} $msg"
}

check_ess_active() {
    [[ -f "$ESS_FLAG_FILE" ]]
}

activate_ess() {
    if check_ess_active; then
        echo -e "${YELLOW}⚠  ESS already active${NC}"
        return 0
    fi

    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  EMERGENCY STOP SYSTEM (ESS) ACTIVATION${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    log_audit "ESS ACTIVATION INITIATED"

    # Set latch flag (prevents auto-restart)
    touch "$ESS_FLAG_FILE"
    chmod 644 "$ESS_FLAG_FILE"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ESS_FLAG_FILE"

    log_audit "ESS LATCH FLAG SET: $ESS_FLAG_FILE"

    # Stop critical trading services
    echo -e "${RED}► Stopping critical trading services...${NC}"
    for service in "${CRITICAL_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service.service" 2>/dev/null; then
            echo "  • Stopping $service..."
            systemctl stop "$service.service" || log_audit "WARNING: Failed to stop $service"
            log_audit "ESS STOPPED SERVICE: $service"
        else
            echo "  • $service (not running)"
        fi
    done

    # Write Redis marker (best-effort)
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping &>/dev/null; then
            redis-cli SET quantum:ess:active "$(date -u +%Y-%m-%dT%H:%M:%SZ)" EX 86400 >/dev/null 2>&1 || true
            redis-cli PUBLISH quantum:event:ess '{"event":"ess_activated","timestamp":'$(date +%s)'}' >/dev/null 2>&1 || true
            log_audit "ESS REDIS MARKER SET"
        fi
    fi

    echo ""
    echo -e "${RED}✓ ESS ACTIVATED${NC}"
    echo -e "${RED}  All trading operations STOPPED${NC}"
    echo -e "${RED}  Monitoring services remain active${NC}"
    echo ""
    echo "To deactivate: $0 deactivate"
    echo ""

    log_audit "ESS ACTIVATION COMPLETE"
}

deactivate_ess() {
    if ! check_ess_active; then
        echo -e "${YELLOW}⚠  ESS not active${NC}"
        return 0
    fi

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  EMERGENCY STOP SYSTEM (ESS) DEACTIVATION${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    log_audit "ESS DEACTIVATION INITIATED"

    # Remove latch flag
    rm -f "$ESS_FLAG_FILE"
    log_audit "ESS LATCH FLAG REMOVED"

    # Clear Redis marker (best-effort)
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping &>/dev/null; then
            redis-cli DEL quantum:ess:active >/dev/null 2>&1 || true
            redis-cli PUBLISH quantum:event:ess '{"event":"ess_deactivated","timestamp":'$(date +%s)'}' >/dev/null 2>&1 || true
            log_audit "ESS REDIS MARKER CLEARED"
        fi
    fi

    # Restart critical services (controlled)
    echo -e "${GREEN}► Restarting critical trading services...${NC}"
    for service in "${CRITICAL_SERVICES[@]}"; do
        if systemctl is-enabled --quiet "$service.service" 2>/dev/null; then
            echo "  • Starting $service..."
            systemctl start "$service.service" || log_audit "WARNING: Failed to start $service"
            log_audit "ESS RESTARTED SERVICE: $service"
        else
            echo "  • $service (disabled, skipping)"
        fi
    done

    echo ""
    echo -e "${GREEN}✓ ESS DEACTIVATED${NC}"
    echo -e "${GREEN}  Trading operations RESUMED${NC}"
    echo ""

    log_audit "ESS DEACTIVATION COMPLETE"
}

status_ess() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  EMERGENCY STOP SYSTEM (ESS) STATUS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if check_ess_active; then
        echo -e "${RED}ESS Status: ACTIVE${NC}"
        echo "Flag file: $ESS_FLAG_FILE"
        echo "Activated: $(cat "$ESS_FLAG_FILE" 2>/dev/null || echo 'UNKNOWN')"
    else
        echo -e "${GREEN}ESS Status: INACTIVE${NC}"
    fi

    echo ""
    echo "Critical Trading Services:"
    for service in "${CRITICAL_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service.service" 2>/dev/null; then
            echo -e "  • $service: ${GREEN}RUNNING${NC}"
        else
            echo -e "  • $service: ${RED}STOPPED${NC}"
        fi
    done

    echo ""
    echo "Monitoring Services:"
    for service in "${MONITOR_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service.service" 2>/dev/null; then
            echo -e "  • $service: ${GREEN}RUNNING${NC}"
        else
            echo -e "  • $service: ${YELLOW}STOPPED${NC}"
        fi
    done

    echo ""
}

main() {
    local cmd="${1:-status}"

    case "$cmd" in
        activate)
            activate_ess
            ;;
        deactivate)
            deactivate_ess
            ;;
        status)
            status_ess
            ;;
        *)
            echo "Usage: $0 {activate|deactivate|status}"
            exit 1
            ;;
    esac
}

main "$@"

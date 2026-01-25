#!/usr/bin/env bash
set -euo pipefail

# P3.4 Reconcile Engine - Deployment & Proof
# Idempotent deployment to VPS

COLOR_BLUE='\033[0;34m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_RESET='\033[0m'

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[✓]${COLOR_RESET} $1"
}

log_error() {
    echo -e "${COLOR_RED}[✗]${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[!]${COLOR_RESET} $1"
}

echo "════════════════════════════════════════════════════════════════"
echo "  P3.4 Reconcile Engine - Deployment & Proof"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if running on VPS
if [[ ! -f "/etc/quantum/apply-layer.env" ]]; then
    log_error "Not on VPS (missing /etc/quantum/apply-layer.env)"
    log_info "This script should run ON the VPS, not locally"
    exit 1
fi

log_info "Step 1: Pull latest code from git..."
cd /opt/quantum || exit 1
git reset --hard
git pull origin main
log_success "Code updated"

log_info "Step 2: Verify P3.4 files exist..."
if [[ ! -f "/opt/quantum/microservices/reconcile_engine/main.py" ]]; then
    log_error "P3.4 main.py not found"
    exit 1
fi
log_success "P3.4 files found"

log_info "Step 3: Copy config files..."
if [[ -f "/opt/quantum/deployment/config/reconcile-engine.env" ]]; then
    cp /opt/quantum/deployment/config/reconcile-engine.env /etc/quantum/
    log_success "Config copied to /etc/quantum/reconcile-engine.env"
else
    log_error "Config file not found in deployment/config/"
    exit 1
fi

log_info "Step 4: Install systemd service..."
if [[ -f "/opt/quantum/deployment/systemd/quantum-reconcile-engine.service" ]]; then
    cp /opt/quantum/deployment/systemd/quantum-reconcile-engine.service /etc/systemd/system/
    systemctl daemon-reload
    log_success "Systemd service installed"
else
    log_error "Systemd service file not found"
    exit 1
fi

log_info "Step 5: Enable and restart P3.4 service..."
systemctl enable quantum-reconcile-engine
systemctl restart quantum-reconcile-engine
sleep 2
log_success "Service restarted"

log_info "Step 6: Also restart P3.3 (for hold check integration)..."
systemctl restart quantum-position-state-brain
sleep 1
log_success "P3.3 restarted"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Deployment Complete - Running Proof Tests"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run proof script
if [[ -f "/opt/quantum/ops/p34_proof.sh" ]]; then
    bash /opt/quantum/ops/p34_proof.sh
else
    log_warn "Proof script not found, running inline checks..."
    
    log_info "Check 1: Service status..."
    if systemctl is-active --quiet quantum-reconcile-engine; then
        log_success "P3.4 service is active"
    else
        log_error "P3.4 service NOT running"
        systemctl status quantum-reconcile-engine --no-pager
        exit 1
    fi
    
    log_info "Check 2: Metrics endpoint..."
    if curl -s http://localhost:8046/metrics | head -3 | grep -q "p34_reconcile"; then
        log_success "Metrics endpoint responding"
    else
        log_error "Metrics endpoint not responding"
        exit 1
    fi
    
    log_info "Check 3: Service logs (last 10 lines)..."
    journalctl -u quantum-reconcile-engine -n 10 --no-pager
    
    log_info "Check 4: Redis state keys..."
    redis-cli KEYS "quantum:reconcile:*" || log_warn "No reconcile keys yet (normal if just started)"
    
    log_success "All basic checks passed"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  P3.4 Reconcile Engine - Deployed Successfully"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor with:"
echo "  journalctl -u quantum-reconcile-engine -f"
echo ""
echo "Check metrics:"
echo "  curl http://localhost:8046/metrics"
echo ""
echo "Check reconcile state:"
echo "  redis-cli HGETALL quantum:reconcile:state:BTCUSDT"
echo ""
echo "Check hold status:"
echo "  redis-cli GET quantum:reconcile:hold:BTCUSDT"
echo ""

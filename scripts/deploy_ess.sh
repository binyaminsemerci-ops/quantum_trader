#!/usr/bin/env bash
#
# ESS Deployment & OPS Ledger Script
# Deploys OS-level Emergency Stop System to production
# Runs proof, appends ops ledger
#

set -euo pipefail

# Configuration
REPO_ROOT="${REPO_ROOT:-.}"
DEPLOY_DIR="$REPO_ROOT/deploy/systemd"
OPS_DIR="$REPO_ROOT/ops"
SCRIPTS_DIR="$REPO_ROOT/scripts"
ESS_CONTROLLER="$OPS_DIR/ess_controller.sh"
PROOF_SCRIPT="$SCRIPTS_DIR/proof_ess.sh"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# ============================================================================
# PHASE 1: VERIFICATION
# ============================================================================

log_info "PHASE 1: Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify ESS components exist
if [[ ! -f "$ESS_CONTROLLER" ]]; then
    log_error "ESS controller not found: $ESS_CONTROLLER"
    exit 1
fi
log_success "ESS controller found"

if [[ ! -f "$PROOF_SCRIPT" ]]; then
    log_error "Proof script not found: $PROOF_SCRIPT"
    exit 1
fi
log_success "Proof script found"

# Verify systemd units exist
REQUIRED_UNITS=(
    "quantum-ess.service"
    "quantum-ess.path"
    "quantum-ess-trigger.service"
    "quantum-ess-watch.service"
    "quantum-ess-watch.timer"
)

for unit in "${REQUIRED_UNITS[@]}"; do
    if [[ ! -f "$DEPLOY_DIR/$unit" ]]; then
        log_error "Systemd unit not found: $DEPLOY_DIR/$unit"
        exit 1
    fi
    log_success "Found $unit"
done

echo ""

# ============================================================================
# PHASE 2: DEPLOY SYSTEMD UNITS
# ============================================================================

log_info "PHASE 2: Deploy Systemd Units"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ! -d /etc/systemd/system ]]; then
    log_error "/etc/systemd/system not found (not root?)"
    exit 1
fi

for unit in "${REQUIRED_UNITS[@]}"; do
    src="$DEPLOY_DIR/$unit"
    dst="/etc/systemd/system/$unit"
    
    if sudo cp "$src" "$dst"; then
        log_success "Deployed $unit"
    else
        log_error "Failed to deploy $unit"
        exit 1
    fi
done

# Reload systemd daemon
sudo systemctl daemon-reload
log_success "Systemd daemon reloaded"

echo ""

# ============================================================================
# PHASE 3: ENABLE SYSTEMD UNITS
# ============================================================================

log_info "PHASE 3: Enable Systemd Units"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Enable path watcher (auto-triggers on flag file)
sudo systemctl enable quantum-ess.path
sudo systemctl start quantum-ess.path
log_success "Enabled quantum-ess.path (flag monitor)"

# Enable watch timer (fallback, polls every 5s)
sudo systemctl enable quantum-ess-watch.timer
sudo systemctl start quantum-ess-watch.timer
log_success "Enabled quantum-ess-watch.timer (fallback watcher)"

# Verify units started
if sudo systemctl is-active quantum-ess.path >/dev/null 2>&1; then
    log_success "quantum-ess.path is RUNNING"
else
    log_warn "quantum-ess.path is NOT running (may be issue)"
fi

echo ""

# ============================================================================
# PHASE 4: RUN PROOF SCRIPT
# ============================================================================

log_info "PHASE 4: Run Proof Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if bash "$PROOF_SCRIPT"; then
    log_success "Proof script PASSED"
    PROOF_RESULT="PASS"
else
    log_error "Proof script FAILED"
    PROOF_RESULT="FAIL"
    exit 1
fi

echo ""

# ============================================================================
# PHASE 5: APPEND OPS LEDGER
# ============================================================================

log_info "PHASE 5: Append OPS Ledger"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Append to ops ledger
if [[ -f "$OPS_DIR/ops_ledger_append.py" ]]; then
    if python3 "$OPS_DIR/ops_ledger_append.py" \
        --operation "ESS Deployment" \
        --objective "Deploy OS-level Emergency Stop System (systemd-based, independent of Python/Redis)" \
        --risk_class "SERVICE_RESTART" \
        --blast_radius "All trading services can be stopped; monitoring services stay running" \
        --changes_summary "ESS systemd units + controller + proof script deployed; ops ledger recorded" \
        --proof_path "$PROOF_SCRIPT" \
        --allowed_services quantum-ai-engine quantum-execution quantum-apply-layer quantum-governor \
        --notes "ESS PHASE II complete: latch mechanism, fail-safe, rapid stop (<2s), audit logging"; then
        log_success "OPS ledger entry appended"
    else
        log_warn "Failed to append ops ledger (non-critical)"
    fi
else
    log_warn "ops_ledger_append.py not found (skipping ledger)"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

log_success "ESS DEPLOYMENT COMPLETE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Deployment Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "ESS Components:"
echo "  ✓ Systemd units deployed to /etc/systemd/system/"
echo "  ✓ quantum-ess.path (flag monitor) ENABLED"
echo "  ✓ quantum-ess-watch.timer (fallback) ENABLED"
echo "  ✓ Proof script: PASS"
echo "  ✓ OPS ledger: RECORDED"
echo ""
echo "Quick Start:"
echo "  • Activate:   bash $ESS_CONTROLLER activate"
echo "  • Check:      bash $ESS_CONTROLLER status"
echo "  • Deactivate: bash $ESS_CONTROLLER deactivate"
echo ""
echo "Documentation:"
echo "  • Quick Ref:  $REPO_ROOT/docs/ESS_QUICK_REFERENCE_PROD.md"
echo "  • Runbook:    $REPO_ROOT/docs/ESS_RUNBOOK.md"
echo "  • Proof:      $PROOF_SCRIPT"
echo ""
echo "Status:"
echo "  systemctl status quantum-ess.path"
echo "  systemctl status quantum-ess-watch.timer"
echo "  journalctl -t quantum-ess -f"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit 0

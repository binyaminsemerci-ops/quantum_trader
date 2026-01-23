#!/usr/bin/env bash
#
# P3 Apply Layer - Deploy Script
#
# Idempotent deployment script for P3 apply layer.
# Copies config, installs systemd unit, enables and starts service.
#
# Usage:
#   ops/p3_deploy.sh
#
# Run from /root/quantum_trader on VPS (will deploy to /home/qt/quantum_trader)
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() {
  echo -e "${GREEN}✅${NC} $1"
}

warn() {
  echo -e "${YELLOW}⚠️ ${NC} $1"
}

fail() {
  echo -e "${RED}❌${NC} $1"
  exit 1
}

info() {
  echo -e "${NC}ℹ️  $1${NC}"
}

echo "=== P3 Apply Layer Deployment ==="
date
echo

# Check running as root
if [[ $EUID -ne 0 ]]; then
  fail "Must run as root (for systemctl and /etc access)"
fi

# Paths
REPO_PATH="/root/quantum_trader"
DEPLOY_PATH="/home/qt/quantum_trader"
ENV_SRC="$REPO_PATH/deployment/config/apply-layer.env"
ENV_DST="/etc/quantum/apply-layer.env"
SYSTEMD_SRC="$REPO_PATH/deployment/systemd/quantum-apply-layer.service"
SYSTEMD_DST="/etc/systemd/system/quantum-apply-layer.service"

# Check repo exists
if [[ ! -d "$REPO_PATH" ]]; then
  fail "Repo not found at $REPO_PATH"
fi

echo "== Step 1: Sync repo to deploy path =="
if [[ ! -d "$DEPLOY_PATH" ]]; then
  info "Creating $DEPLOY_PATH"
  mkdir -p "$DEPLOY_PATH"
  chown qt:qt "$DEPLOY_PATH"
fi

info "Syncing $REPO_PATH -> $DEPLOY_PATH"
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'venv' \
  --exclude '.venv' \
  "$REPO_PATH/" "$DEPLOY_PATH/"

chown -R qt:qt "$DEPLOY_PATH"
ok "Repo synced to $DEPLOY_PATH"

echo

echo "== Step 2: Install config file (idempotent) =="
if [[ ! -f "$ENV_DST" ]]; then
  info "Creating /etc/quantum directory"
  mkdir -p /etc/quantum
  
  info "Copying $ENV_SRC -> $ENV_DST"
  cp "$ENV_SRC" "$ENV_DST"
  chmod 640 "$ENV_DST"
  chown root:qt "$ENV_DST"
  
  ok "Config installed: $ENV_DST"
  warn "IMPORTANT: Edit $ENV_DST to add Binance testnet credentials if needed"
else
  info "Config already exists: $ENV_DST (not overwriting)"
  info "Current mode: $(grep APPLY_MODE $ENV_DST)"
fi

echo

echo "== Step 3: Install systemd unit =="
info "Copying $SYSTEMD_SRC -> $SYSTEMD_DST"
cp "$SYSTEMD_SRC" "$SYSTEMD_DST"
chmod 644 "$SYSTEMD_DST"

ok "Systemd unit installed: $SYSTEMD_DST"

echo

echo "== Step 4: Enable and start service =="
info "Running systemctl daemon-reload"
systemctl daemon-reload

info "Enabling quantum-apply-layer"
systemctl enable quantum-apply-layer

info "Starting quantum-apply-layer"
systemctl restart quantum-apply-layer

# Wait for service to start
sleep 2

if systemctl is-active --quiet quantum-apply-layer; then
  ok "Service active"
else
  fail "Service failed to start (check logs below)"
fi

echo

echo "== Step 5: Service status =="
systemctl status quantum-apply-layer --no-pager || true

echo

echo "== Step 6: Recent logs =="
journalctl -u quantum-apply-layer -n 20 --no-pager

echo

echo "== Step 7: Verify Redis streams =="
info "Waiting 10s for first cycle..."
sleep 10

plan_count=$(redis-cli XLEN quantum:stream:apply.plan 2>/dev/null || echo "0")
result_count=$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null || echo "0")

echo "  apply.plan: $plan_count entries"
echo "  apply.result: $result_count entries"

if [[ $plan_count -gt 0 ]]; then
  ok "Apply plans being created"
else
  warn "No plans yet (check harvest proposals exist)"
fi

echo

echo "== Step 8: Verify Prometheus metrics =="
if curl -s http://localhost:8043/metrics >/dev/null 2>&1; then
  ok "Prometheus metrics available on :8043"
  info "Sample metrics:"
  curl -s http://localhost:8043/metrics | grep quantum_apply | head -5
else
  warn "Prometheus metrics not yet available (may need more time)"
fi

echo

echo "=== Deployment Complete ==="
echo
info "Summary:"
echo "  - Service: $(systemctl is-active quantum-apply-layer)"
echo "  - Config: $ENV_DST"
echo "  - Mode: $(grep APPLY_MODE $ENV_DST | cut -d= -f2)"
echo "  - Allowlist: $(grep APPLY_ALLOWLIST $ENV_DST | cut -d= -f2)"
echo "  - Plans: $plan_count"
echo "  - Results: $result_count"
echo
ok "P3 Apply Layer deployed successfully"
echo
info "Next steps:"
echo "  1. Run proof pack: bash $DEPLOY_PATH/ops/p3_proof_dry_run.sh"
echo "  2. Monitor logs: journalctl -u quantum-apply-layer -f"
echo "  3. Check metrics: curl http://localhost:8043/metrics | grep quantum_apply"
echo "  4. For testnet: edit $ENV_DST (add credentials, set APPLY_MODE=testnet), then systemctl restart quantum-apply-layer"

#!/usr/bin/env bash
# P2.6 Portfolio Gate - Deployment & Proof
#
# WHAT THIS DOES:
# - Deploys P2.6 Portfolio Gate service to VPS
# - Updates Apply Layer to require P2.6 permits
# - Verifies all components operational
# - Runs proof pack to validate integration
#
# FAIL-CLOSED: Any error → abort deployment

set -euo pipefail

echo "=============================="
echo "P2.6 DEPLOYMENT STARTING"
echo "=============================="
echo ""

# Configuration
VPS_ROOT="/root/quantum_trader"
VPS_WORK="/home/qt/quantum_trader"
SERVICE_NAME="quantum-portfolio-gate"
CONFIG_FILE="portfolio-gate.env"
PROOF_OUTPUT="${VPS_WORK}/docs/P2_6_VPS_PROOF.txt"

# Step 1: Pull latest code in root
echo "[1/10] Git pull in ${VPS_ROOT}..."
cd "${VPS_ROOT}" || exit 1
git pull origin main
echo "✓ Code updated"
echo ""

# Step 2: Rsync to working directory
echo "[2/10] Rsync to ${VPS_WORK}..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='logs/' \
    "${VPS_ROOT}/" "${VPS_WORK}/"
chown -R qt:qt "${VPS_WORK}"
echo "✓ Files synced"
echo ""

# Step 3: Install P2.6 config
echo "[3/10] Installing /etc/quantum/${CONFIG_FILE}..."
mkdir -p /etc/quantum

cp "${VPS_WORK}/deployment/config/${CONFIG_FILE}" "/etc/quantum/${CONFIG_FILE}"
echo "✓ P2.6 config installed"
echo ""

# Step 4: Install systemd service
echo "[4/10] Installing systemd service..."
cp "${VPS_WORK}/deployment/systemd/${SERVICE_NAME}.service" "/etc/systemd/system/"
systemctl daemon-reload
echo "✓ Systemd service installed"
echo ""

# Step 5: Enable and start P2.6 service
echo "[5/10] Starting ${SERVICE_NAME}..."
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
sleep 3

# Check if service started
if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "✓ ${SERVICE_NAME} ACTIVE"
else
    echo "❌ FAILED: ${SERVICE_NAME} not active"
    systemctl status "${SERVICE_NAME}" --no-pager || true
    exit 1
fi
echo ""

# Step 6: Check metrics endpoint
echo "[6/10] Checking P2.6 metrics (port 8047)..."
METRICS=$(curl -s http://localhost:8047/metrics | grep "p26_" || echo "NO_METRICS")
if echo "${METRICS}" | grep -q "p26_"; then
    echo "✓ P2.6 metrics responding"
else
    echo "❌ WARNING: P2.6 metrics not found"
fi
echo ""

# Step 7: Restart Apply Layer (updated to require P2.6 permit)
echo "[7/10] Restarting quantum-apply-layer (P2.6 permit now required)..."
systemctl restart quantum-apply-layer
sleep 2

if systemctl is-active --quiet quantum-apply-layer; then
    echo "✓ Apply Layer restarted successfully"
else
    echo "❌ FAILED: Apply Layer not active after restart"
    systemctl status quantum-apply-layer --no-pager || true
    exit 1
fi
echo ""

# Step 8: Check stream connectivity
echo "[8/10] Checking Redis streams..."
HARVEST_LEN=$(redis-cli XLEN quantum:stream:harvest.proposal || echo "0")
GATE_LEN=$(redis-cli XLEN quantum:stream:portfolio.gate || echo "0")
echo "  harvest.proposal stream length: ${HARVEST_LEN}"
echo "  portfolio.gate stream length: ${GATE_LEN}"
echo ""

# Step 9: Check for P2.6 permits
echo "[9/10] Checking P2.6 permits..."
P26_PERMITS=$(redis-cli --scan --pattern 'quantum:permit:p26:*' | wc -l || echo "0")
echo "  P2.6 permits found: ${P26_PERMITS}"
echo ""

# Step 10: Generate proof document
echo "[10/10] Generating proof document..."
{
    echo "====================================="
    echo "P2.6 PORTFOLIO GATE - VPS PROOF"
    echo "Timestamp: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "====================================="
    echo ""
    
    echo "===== SERVICE STATUS ====="
    systemctl is-active quantum-portfolio-gate quantum-apply-layer || true
    echo ""
    
    echo "===== P2.6 METRICS (first 40 lines) ====="
    curl -s http://localhost:8047/metrics | head -40
    echo ""
    
    echo "===== PORTFOLIO.GATE STREAM INFO ====="
    redis-cli XINFO STREAM quantum:stream:portfolio.gate | head -30 || echo "Stream may not exist yet (OK if no proposals processed)"
    echo ""
    
    echo "===== P2.6 PERMITS ====="
    redis-cli --scan --pattern 'quantum:permit:p26:*' | head -10
    echo ""
    
    echo "===== P2.6 LOGS (last 80 lines) ====="
    journalctl -u quantum-portfolio-gate --since "2 minutes ago" -n 80 --no-pager
    echo ""
    
    echo "===== APPLY LAYER LOGS (last 40 lines, P2.6 integration check) ====="
    journalctl -u quantum-apply-layer --since "2 minutes ago" -n 40 --no-pager | grep -i "permit\|p26\|p2.6" || echo "No P2.6 mentions yet (OK if no execution attempts)"
    echo ""
    
    echo "===== PROOF COMPLETE ====="
    echo "Review: cat ${PROOF_OUTPUT}"
    
} > "${PROOF_OUTPUT}"

chown qt:qt "${PROOF_OUTPUT}"

echo "✓ Proof document saved: ${PROOF_OUTPUT}"
echo ""
echo "=============================="
echo "P2.6 DEPLOYMENT COMPLETE ✅"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Review proof: cat ${PROOF_OUTPUT}"
echo "2. Monitor P2.6: journalctl -u quantum-portfolio-gate -f"
echo "3. Monitor Apply Layer: journalctl -u quantum-apply-layer -f"
echo "4. Check metrics: curl http://localhost:8047/metrics"
echo ""
echo "Rollback (if needed):"
echo "  systemctl stop quantum-portfolio-gate"
echo "  systemctl disable quantum-portfolio-gate"
echo "  git revert <commit>"
echo "  rsync + systemctl restart quantum-apply-layer"
echo ""

# Exit 0 on success (CI/ops best practice)
if systemctl is-active --quiet quantum-portfolio-gate && \
   curl -sf http://127.0.0.1:8047/metrics > /dev/null && \
   [ -f "${PROOF_OUTPUT}" ]; then
    echo "✅ Deployment verified - exit 0"
    exit 0
else
    echo "❌ Deployment verification failed - exit 1"
    exit 1
fi

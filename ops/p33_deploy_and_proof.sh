#!/usr/bin/env bash
# P3.3 Position State Brain - Deployment & Proof
# 
# WHAT THIS DOES:
# - Deploys P3.3 Position State Brain service to VPS
# - Integrates Apply Layer with P3.3 permit checks
# - Verifies all components operational
# - Runs proof pack to validate sanity checks
#
# FAIL-CLOSED: Any error → abort deployment

set -euo pipefail

echo "=============================="
echo "P3.3 DEPLOYMENT STARTING"
echo "=============================="
echo ""

# Configuration
VPS_ROOT="/root/quantum_trader"
VPS_WORK="/home/qt/quantum_trader"
SERVICE_NAME="quantum-position-state-brain"
CONFIG_FILE="position-state-brain.env"
PROOF_OUTPUT="${VPS_WORK}/docs/P3_3_VPS_PROOF.txt"

# Step 1: Pull latest code in root
echo "[1/9] Git pull in ${VPS_ROOT}..."
cd "${VPS_ROOT}" || exit 1
git pull origin main
echo "✓ Code updated"
echo ""

# Step 2: Rsync to working directory
echo "[2/9] Rsync to ${VPS_WORK}..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='logs/' \
    "${VPS_ROOT}/" "${VPS_WORK}/"
echo "✓ Files synced"
echo ""

# Step 3: Install P3.3 config
echo "[3/9] Installing /etc/quantum/${CONFIG_FILE}..."
mkdir -p /etc/quantum

# Copy template
cp "${VPS_WORK}/deployment/config/${CONFIG_FILE}" "/etc/quantum/${CONFIG_FILE}"

# Copy Binance credentials from testnet.env
if [ -f "/etc/quantum/testnet.env" ]; then
    BINANCE_KEY=$(grep "^BINANCE_API_KEY=" /etc/quantum/testnet.env | cut -d'=' -f2-)
    BINANCE_SECRET=$(grep "^BINANCE_API_SECRET=" /etc/quantum/testnet.env | cut -d'=' -f2-)
    
    if [ -n "${BINANCE_KEY}" ] && [ -n "${BINANCE_SECRET}" ]; then
        echo "BINANCE_API_KEY=${BINANCE_KEY}" >> "/etc/quantum/${CONFIG_FILE}"
        echo "BINANCE_API_SECRET=${BINANCE_SECRET}" >> "/etc/quantum/${CONFIG_FILE}"
        echo "✓ Binance credentials copied"
    else
        echo "⚠ WARNING: Binance credentials not found in testnet.env"
    fi
else
    echo "⚠ WARNING: /etc/quantum/testnet.env not found"
fi
echo ""

# Step 4: Install systemd service
echo "[4/9] Installing systemd service..."
cp "${VPS_WORK}/deployment/systemd/${SERVICE_NAME}.service" "/etc/systemd/system/"
systemctl daemon-reload
echo "✓ Systemd service installed"
echo ""

# Step 5: Enable and start P3.3 service
echo "[5/9] Starting ${SERVICE_NAME}..."
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
echo "[6/9] Checking P3.3 metrics (port 8045)..."
METRICS=$(curl -s http://localhost:8045/metrics | grep "p33_" || echo "NO_METRICS")
if echo "${METRICS}" | grep -q "p33_"; then
    echo "✓ P3.3 metrics responding"
else
    echo "⚠ WARNING: P3.3 metrics not found yet (service may need warmup)"
fi
echo ""

# Step 7: Restart Apply Layer (to pick up P3.3 integration)
echo "[7/9] Restarting Apply Layer..."
systemctl restart quantum-apply-layer
sleep 2

if systemctl is-active --quiet quantum-apply-layer; then
    echo "✓ Apply Layer restarted"
else
    echo "❌ FAILED: Apply Layer not active"
    systemctl status quantum-apply-layer --no-pager || true
    exit 1
fi
echo ""

# Step 8: Run proof script
echo "[8/9] Running proof pack..."
mkdir -p "${VPS_WORK}/docs"
bash "${VPS_WORK}/ops/p33_proof.sh" > "${PROOF_OUTPUT}" 2>&1
echo "✓ Proof completed → ${PROOF_OUTPUT}"
echo ""

# Step 9: Summary
echo "[9/9] Deployment summary:"
echo "  - P3.3 service: $(systemctl is-active ${SERVICE_NAME})"
echo "  - Apply Layer: $(systemctl is-active quantum-apply-layer)"
echo "  - Governor: $(systemctl is-active quantum-governor)"
echo "  - Proof pack: ${PROOF_OUTPUT}"
echo ""

# Show first 20 lines of proof
echo "=============================="
echo "PROOF PREVIEW (first 20 lines):"
echo "=============================="
head -20 "${PROOF_OUTPUT}"
echo ""
echo "Full proof: ${PROOF_OUTPUT}"
echo ""

echo "=============================="
echo "P3.3 DEPLOYMENT COMPLETE ✓"
echo "=============================="

#!/bin/bash
# P2.6 Portfolio Heat Gate - VPS Deployment Script
# Idempotent, fail-closed, production-grade

set -euo pipefail

echo "=============================================="
echo "P2.6 Portfolio Heat Gate - Deployment"
echo "=============================================="

# Configuration
SERVICE_NAME="quantum-portfolio-heat-gate"
CODE_DIR="/home/qt/quantum_trader"
SYSTEMD_DIR="/etc/systemd/system"
ENV_DIR="/etc/quantum"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
   echo "ERROR: Must run as root"
   exit 1
fi

echo "[1/6] Creating directory structure..."
mkdir -p ${CODE_DIR}/microservices/portfolio_heat_gate
mkdir -p ${ENV_DIR}
chown -R qt:qt ${CODE_DIR}

echo "[2/6] Installing service code..."
# Code should be synced separately via rsync/scp
if [ ! -f "${CODE_DIR}/microservices/portfolio_heat_gate/main.py" ]; then
    echo "ERROR: Service code not found. Please sync code first."
    exit 1
fi
chmod +x ${CODE_DIR}/microservices/portfolio_heat_gate/main.py
chown -R qt:qt ${CODE_DIR}/microservices/portfolio_heat_gate

echo "[3/6] Installing environment file..."
if [ ! -f "${ENV_DIR}/portfolio-heat-gate.env" ]; then
    if [ -f "${CODE_DIR}/deploy/env/portfolio-heat-gate.env" ]; then
        cp ${CODE_DIR}/deploy/env/portfolio-heat-gate.env ${ENV_DIR}/portfolio-heat-gate.env
    else
        echo "ERROR: Environment file not found"
        exit 1
    fi
fi
chmod 640 ${ENV_DIR}/portfolio-heat-gate.env
chown root:qt ${ENV_DIR}/portfolio-heat-gate.env

echo "[4/6] Installing systemd service..."
if [ -f "${CODE_DIR}/deploy/systemd/${SERVICE_NAME}.service" ]; then
    cp ${CODE_DIR}/deploy/systemd/${SERVICE_NAME}.service ${SYSTEMD_DIR}/
    chmod 644 ${SYSTEMD_DIR}/${SERVICE_NAME}.service
else
    echo "ERROR: Service file not found"
    exit 1
fi

echo "[5/6] Reloading systemd..."
systemctl daemon-reload

echo "[6/6] Starting service..."
systemctl enable ${SERVICE_NAME}
systemctl restart ${SERVICE_NAME}

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Service status:"
systemctl status ${SERVICE_NAME} --no-pager | head -15
echo ""
echo "Check logs:"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "Metrics:"
echo "  curl http://localhost:8056/metrics"
echo ""

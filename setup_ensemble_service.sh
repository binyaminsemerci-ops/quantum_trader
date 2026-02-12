#!/bin/bash
# Setup Ensemble Predictor Service with Dependencies
#
# This script:
# 1. Creates Python venv
# 2. Installs required dependencies
# 3. Verifies service can start
# 4. Starts the service

set -e

SCRIPT_DIR="/home/qt/quantum_trader"
VENV_DIR="$SCRIPT_DIR/venv"

echo "============================================================================"
echo "Setting up Ensemble Predictor Service"
echo "============================================================================"
echo ""

# Check if running as root (need to switch to qt user)
if [ "$EUID" -eq 0 ]; then
    echo "Running as root, will create venv as qt user..."
    SUDO_CMD="sudo -u qt"
else
    SUDO_CMD=""
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating Python virtual environment..."
    cd $SCRIPT_DIR
    $SUDO_CMD python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "[1/5] Virtual environment already exists"
fi

# Activate venv and install dependencies
echo ""
echo "[2/5] Installing dependencies..."

# Use the venv's pip to install packages
$SUDO_CMD $VENV_DIR/bin/pip install --upgrade pip

# Core dependencies for ensemble predictor
$SUDO_CMD $VENV_DIR/bin/pip install \
    redis \
    aioredis \
    scikit-learn \
    numpy \
    pandas

echo "✅ Dependencies installed"

# Verify imports
echo ""
echo "[3/5] Verifying imports..."

$SUDO_CMD $VENV_DIR/bin/python -c "
import redis.asyncio as aioredis
import sklearn
import numpy as np
import pandas as pd
print('✅ All imports successful')
print(f'  sklearn: {sklearn.__version__}')
print(f'  numpy: {np.__version__}')
print(f'  pandas: {pd.__version__}')
"

# Fix file permissions (ensure qt user can execute)
echo ""
echo "[4/5] Setting file permissions..."
chown -R qt:qt $SCRIPT_DIR/ai_engine
chmod +x $SCRIPT_DIR/ai_engine/services/ensemble_predictor_service.py
echo "✅ Permissions set"

# Test service can start (dry run)
echo ""
echo "[5/5] Testing service startup (3 second test)..."

# Start service in background for quick test
$SUDO_CMD timeout 3 $VENV_DIR/bin/python $SCRIPT_DIR/ai_engine/services/ensemble_predictor_service.py 2>&1 | head -20 || true

echo ""
echo "============================================================================"
echo "✅ Setup Complete"
echo "============================================================================"
echo ""
echo "Now starting service with systemd..."
systemctl daemon-reload
systemctl restart quantum-ensemble-predictor.service
sleep 3

if systemctl is-active --quiet quantum-ensemble-predictor.service; then
    echo "✅ Service is ACTIVE"
    echo ""
    echo "Service status:"
    systemctl status quantum-ensemble-predictor.service --no-pager -n 0
    echo ""
    echo "Recent logs:"
    journalctl -u quantum-ensemble-predictor.service -n 20 --no-pager
else
    echo "❌ Service failed to start"
    echo ""
    echo "Service status:"
    systemctl status quantum-ensemble-predictor.service --no-pager
    echo ""
    echo "Recent logs:"
    journalctl -u quantum-ensemble-predictor.service -n 50 --no-pager
    exit 1
fi

echo ""
echo "✅ Ensemble Predictor Service is running!"
echo ""
echo "Monitor service:"
echo "  systemctl status quantum-ensemble-predictor.service"
echo "  journalctl -u quantum-ensemble-predictor.service -f"
echo ""
echo "Check signal production:"
echo "  redis-cli XLEN quantum:stream:signal.score"
echo "  redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 5"
echo ""

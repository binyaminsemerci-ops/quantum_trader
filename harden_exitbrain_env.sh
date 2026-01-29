#!/bin/bash
# P0.HARDEN-EXITBRAIN-ENV
# Creates dedicated environment file for Exit Brain v3.5

set -euo pipefail

echo "=== P0.HARDEN-EXITBRAIN-ENV ==="
echo ""

# Extract credentials from testnet.env (source of truth)
echo "1) Reading credentials from /etc/quantum/testnet.env..."
API_KEY=$(grep "^BINANCE_API_KEY=" /etc/quantum/testnet.env | cut -d= -f2)
API_SECRET=$(grep "^BINANCE_API_SECRET=" /etc/quantum/testnet.env | cut -d= -f2)

if [ -z "$API_KEY" ] || [ -z "$API_SECRET" ]; then
    echo "❌ Failed to extract credentials from testnet.env"
    exit 1
fi

echo "✅ Credentials extracted (length: ${#API_KEY} chars)"
echo ""

# Create dedicated exitbrain-v35.env
echo "2) Creating /etc/quantum/exitbrain-v35.env..."
cat > /etc/quantum/exitbrain-v35.env << EOF
# ============================================
# Exit Brain v3.5 Environment Configuration
# ============================================
# Source: /etc/quantum/testnet.env (credentials)
# DO NOT edit credentials here - use testnet.env as source of truth
# To rotate keys: update testnet.env, then re-run this script

# Binance Testnet Credentials (clean names for backend code)
BINANCE_API_KEY=${API_KEY}
BINANCE_API_SECRET=${API_SECRET}
BINANCE_TESTNET=true
USE_BINANCE_TESTNET=true

# Exit Brain v3.5 Configuration
EXIT_BRAIN_V35_ENABLED=true
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=LIVE
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED

# Kill-Switch (fail-closed safety)
# Set to "true" to force shadow mode regardless of other settings
# Service restart required after changing this
EXIT_EXECUTOR_KILL_SWITCH=false

# Python Environment
PYTHONUNBUFFERED=1
EOF

chmod 600 /etc/quantum/exitbrain-v35.env
chown root:root /etc/quantum/exitbrain-v35.env

echo "✅ Created /etc/quantum/exitbrain-v35.env"
echo ""

# Show file info
echo "3) File permissions:"
ls -la /etc/quantum/exitbrain-v35.env
echo ""

# Show variables (redacted)
echo "4) Variables configured (credentials redacted):"
grep -E "^[A-Z_]+=" /etc/quantum/exitbrain-v35.env | sed 's/\(KEY\|SECRET\)=.*/\1=***REDACTED***/'
echo ""

# Update service file
echo "5) Updating quantum-exitbrain-v35.service..."
cat > /etc/systemd/system/quantum-exitbrain-v35.service << 'SERVICEEOF'
[Unit]
Description=Quantum Exit Brain v3.5 (TESTNET)
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/exitbrain-v35.env
Environment=PYTHONPATH=/home/qt/quantum_trader:/home/qt/quantum_trader/microservices
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 microservices/position_monitor/main_exitbrain.py
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/quantum/exitbrain_v35.log
StandardError=append:/var/log/quantum/exitbrain_v35.log

[Install]
WantedBy=multi-user.target
SERVICEEOF

echo "✅ Service file updated"
echo ""

# Reload and restart
echo "6) Reloading systemd and restarting service..."
systemctl daemon-reload
systemctl restart quantum-exitbrain-v35.service
sleep 8

# Verify
echo "7) Service status:"
systemctl is-active quantum-exitbrain-v35.service && echo "✅ Service ACTIVE" || echo "❌ Service FAILED"
echo ""

# Check logs for LIVE mode
echo "8) Checking for LIVE mode activation..."
tail -100 /var/log/quantum/exitbrain_v35.log | grep -E "LIVE MODE|EXIT_MODE=EXIT_BRAIN_V3" | tail -5
echo ""

echo "=== P0.HARDEN COMPLETE ==="
echo ""
echo "Summary:"
echo "  - Dedicated env file: /etc/quantum/exitbrain-v35.env"
echo "  - Source of truth: /etc/quantum/testnet.env (unchanged)"
echo "  - Kill-switch: EXIT_EXECUTOR_KILL_SWITCH=false (active)"
echo "  - Service: quantum-exitbrain-v35.service (restarted)"
echo ""
echo "To activate kill-switch:"
echo "  sed -i 's/EXIT_EXECUTOR_KILL_SWITCH=false/EXIT_EXECUTOR_KILL_SWITCH=true/' /etc/quantum/exitbrain-v35.env"
echo "  systemctl restart quantum-exitbrain-v35.service"

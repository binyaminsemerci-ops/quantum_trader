#!/bin/bash
set -e

echo "=== Deploying P1.5 Risk Proposal Publisher ==="
echo "Target: VPS (systemd)"
echo ""

# Check we're on VPS
if [ ! -f /etc/systemd/system/quantum-marketstate.service ]; then
    echo "❌ ERROR: Not on VPS or quantum-marketstate.service not found"
    echo "Run this script on the VPS where quantum-marketstate is deployed"
    exit 1
fi

echo "✅ VPS environment detected"
echo ""

# Create config directory if needed
sudo mkdir -p /etc/quantum
echo "✅ Config directory ready: /etc/quantum"

# Copy env file
echo "Copying configuration..."
sudo cp deployment/config/risk-proposal.env /etc/quantum/risk-proposal.env
sudo chown qt:qt /etc/quantum/risk-proposal.env
sudo chmod 640 /etc/quantum/risk-proposal.env
echo "✅ Config file: /etc/quantum/risk-proposal.env"

# Copy systemd service
echo "Copying systemd service..."
sudo cp deployment/systemd/quantum-risk-proposal.service /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/quantum-risk-proposal.service
echo "✅ Service file: /etc/systemd/system/quantum-risk-proposal.service"

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload
echo "✅ systemd reloaded"

# Enable and start service
echo ""
echo "Enabling and starting service..."
sudo systemctl enable quantum-risk-proposal.service
sudo systemctl restart quantum-risk-proposal.service

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service status:"
sudo systemctl status quantum-risk-proposal.service --no-pager -l || true

echo ""
echo "=== Verification Commands ==="
echo ""
echo "1. Check service status:"
echo "   sudo systemctl status quantum-risk-proposal"
echo ""
echo "2. View logs (last 50 lines):"
echo "   sudo journalctl -u quantum-risk-proposal -n 50 --no-pager"
echo ""
echo "3. Follow logs in real-time:"
echo "   sudo journalctl -u quantum-risk-proposal -f"
echo ""
echo "4. Check Redis proposals:"
echo "   redis-cli HGETALL quantum:risk:proposal:BTCUSDT"
echo "   redis-cli HGETALL quantum:risk:proposal:ETHUSDT"
echo "   redis-cli HGETALL quantum:risk:proposal:SOLUSDT"
echo ""
echo "5. Test single cycle (debug mode):"
echo "   cd /home/qt/quantum_trader"
echo "   source /opt/quantum/venvs/ai-engine/bin/activate"
echo "   python3 microservices/risk_proposal_publisher/main.py --once"
echo ""
echo "6. Stop service:"
echo "   sudo systemctl stop quantum-risk-proposal"
echo ""
echo "7. Restart service:"
echo "   sudo systemctl restart quantum-risk-proposal"
echo ""

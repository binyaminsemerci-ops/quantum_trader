#!/bin/bash
set -e

echo "=== Deploying P2.5 Harvest Proposal Publisher ==="
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

# Check dependencies
echo "Checking dependencies..."
if [ ! -f /etc/systemd/system/quantum-marketstate.service ]; then
    echo "⚠️  WARNING: quantum-marketstate.service not found (P0.5 MarketState publisher)"
fi
if [ ! -f /etc/systemd/system/quantum-risk-proposal.service ]; then
    echo "⚠️  WARNING: quantum-risk-proposal.service not found (P1.5 Risk Proposal publisher)"
fi

# Create config directory if needed
sudo mkdir -p /etc/quantum
echo "✅ Config directory ready: /etc/quantum"

# Copy env file
echo "Copying configuration..."
sudo cp deployment/config/harvest-proposal.env /etc/quantum/harvest-proposal.env
sudo chown qt:qt /etc/quantum/harvest-proposal.env
sudo chmod 640 /etc/quantum/harvest-proposal.env
echo "✅ Config file: /etc/quantum/harvest-proposal.env"

# Copy systemd service
echo "Copying systemd service..."
sudo cp deployment/systemd/quantum-harvest-proposal.service /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/quantum-harvest-proposal.service
echo "✅ Service file: /etc/systemd/system/quantum-harvest-proposal.service"

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload
echo "✅ systemd reloaded"

# Enable and start service
echo ""
echo "Enabling and starting service..."
sudo systemctl enable quantum-harvest-proposal.service
sudo systemctl restart quantum-harvest-proposal.service

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service status:"
sudo systemctl status quantum-harvest-proposal.service --no-pager -l || true

echo ""
echo "=== Verification Commands ==="
echo ""
echo "1. Check service status:"
echo "   sudo systemctl status quantum-harvest-proposal"
echo ""
echo "2. View logs (last 50 lines):"
echo "   sudo journalctl -u quantum-harvest-proposal -n 50 --no-pager"
echo ""
echo "3. Follow logs in real-time:"
echo "   sudo journalctl -u quantum-harvest-proposal -f"
echo ""
echo "4. Check Redis harvest proposals:"
echo "   redis-cli HGETALL quantum:harvest:proposal:BTCUSDT"
echo "   redis-cli HGETALL quantum:harvest:proposal:ETHUSDT"
echo "   redis-cli HGETALL quantum:harvest:proposal:SOLUSDT"
echo ""
echo "5. Test single cycle (debug mode):"
echo "   cd /home/qt/quantum_trader"
echo "   source /opt/quantum/venvs/ai-engine/bin/activate"
echo "   python3 microservices/harvest_proposal_publisher/main.py --once --position-source synthetic"
echo ""
echo "6. Check harvest action distribution:"
echo "   redis-cli HGET quantum:harvest:proposal:BTCUSDT harvest_action"
echo "   redis-cli HGET quantum:harvest:proposal:BTCUSDT kill_score"
echo "   redis-cli HGET quantum:harvest:proposal:BTCUSDT R_net"
echo ""
echo "7. Stop service:"
echo "   sudo systemctl stop quantum-harvest-proposal"
echo ""
echo "8. Restart service:"
echo "   sudo systemctl restart quantum-harvest-proposal"
echo ""
echo "=== Next Steps ==="
echo ""
echo "Run P2.5B proof pack to verify deployment:"
echo "  1. systemctl status quantum-harvest-proposal (service active)"
echo "  2. Calc-only hygiene (grep + AST imports)"
echo "  3. P0.5 MarketState data exists"
echo "  4. P1.5 proposals exist (or fallback works)"
echo "  5. P2.5 harvest proposals published"
echo "  6. --once mode works"
echo "  7. Rate limiting (10s interval)"
echo ""

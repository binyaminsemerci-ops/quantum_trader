#!/bin/bash
# Deploy Service Health Fixes to VPS
# Fixes: Risk Safety path + Meta Regime data stream

set -e

echo "🚀 DEPLOYING SERVICE HEALTH FIXES"
echo "===================================="
echo ""
echo "📋 Changes:"
echo "   1. Risk Safety: Fixed service configuration"
echo "   2. Meta Regime: Read from quantum:stream:exchange.raw"
echo "   3. Portfolio Governance: No change (0 samples is normal)"
echo ""

cd /root/quantum_trader

echo "📥 Pulling latest code..."
git pull origin main

echo ""
echo "🛑 Stopping affected services..."
sudo systemctl stop quantum-risk-safety.service quantum-meta-regime.service

echo ""
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "🚀 Starting services..."
sudo systemctl start quantum-risk-safety.service quantum-meta-regime.service

echo ""
echo "⏳ Waiting 15 seconds for startup..."
sleep 15

echo ""
echo "🔍 VERIFICATION"
echo "=============="

echo ""
echo "1️⃣ Risk Safety Status:"
sudo systemctl status quantum-risk-safety.service --no-pager | head -10

echo ""
echo "2️⃣ Meta Regime Status:"
sudo systemctl status quantum-meta-regime.service --no-pager | head -10

echo ""
echo "3️⃣ Risk Safety Logs (last 5):"
sudo journalctl -u quantum-risk-safety.service -n 5 --no-pager

echo ""
echo "4️⃣ Meta Regime Logs (last 5):"
sudo journalctl -u quantum-meta-regime.service -n 5 --no-pager

echo ""
echo "5️⃣ Checking for market data processing..."
sleep 30
REGIME_LOGS=$(sudo journalctl -u quantum-meta-regime.service -n 10 --no-pager)
if echo "$REGIME_LOGS" | grep -q "regime_detected\|samples"; then
    echo "✅ Meta Regime processing data successfully!"
else
    echo "⚠️  Meta Regime still warming up..."
fi

echo ""
echo "✅ DEPLOYMENT COMPLETE"
echo ""
echo "📊 Full system check:"
systemctl list-units 'quantum-risk-safety.service' 'quantum-meta-regime.service' 'quantum-portfolio-governance.service' --no-pager --no-legend

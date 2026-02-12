#!/bin/bash
set -e

echo "🎲 Enabling SYNTHETIC MODE for ensemble predictor..."

cd /home/qt/quantum_trader

# Update systemd service to add SYNTHETIC_MODE environment variable
SERVICE_FILE="/etc/systemd/system/quantum-ensemble-predictor.service"

echo "📝 Adding SYNTHETIC_MODE=true to service environment..."
sudo sed -i '/Environment="REDIS_URL=/a Environment="SYNTHETIC_MODE=true"' "$SERVICE_FILE"

echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "🔄 Restarting ensemble predictor..."
sudo systemctl restart quantum-ensemble-predictor.service

sleep 5

echo "📊 Service status:"
sudo systemctl status quantum-ensemble-predictor.service --no-pager -n 30 | tail -20

echo ""
echo "✅ Synthetic mode enabled!"
echo "🎯 Predictor will now generate random confidences (0.25-0.75) for calibration testing"
echo ""
echo "📈 Check signals with:"
echo "redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 5"

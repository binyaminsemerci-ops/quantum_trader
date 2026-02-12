#!/bin/bash
set -e

echo "🔧 Disabling SYNTHETIC_MODE and enabling real models..."

# Remove SYNTHETIC_MODE from service file
sudo sed -i '/SYNTHETIC_MODE/d' /etc/systemd/system/quantum-ensemble-predictor.service

# Reload systemd
sudo systemctl daemon-reload

# Clear Python cache
find /home/qt/quantum_trader -name "*.pyc" -delete 2>/dev/null || true

# Restart service
sudo systemctl restart quantum-ensemble-predictor.service

sleep 10

echo "📊 Service status:"
sudo systemctl status quantum-ensemble-predictor.service --no-pager -n 30 | tail -20

echo ""
echo "🔍 Checking for model loading..."
sudo journalctl -u quantum-ensemble-predictor.service -n 50 --no-pager | grep -E "Models loaded|Model loading failed|ENSEMBLE-PREDICTOR.*✅|confidence" | tail -10

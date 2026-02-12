#!/bin/bash
set -e

echo "🔧 Installing PyTorch (CPU-only) for ensemble predictor..."

cd /home/qt/quantum_trader

# Activate venv
source venv/bin/activate

# Install CPU-only PyTorch (lighter, faster for this use case)
echo "📦 Installing torch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
echo "✅ Verifying torch import..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Restart ensemble predictor
echo "🔄 Restarting ensemble predictor service..."
sudo systemctl restart quantum-ensemble-predictor.service

sleep 3

# Check status
echo "📊 Service status:"
sudo systemctl status quantum-ensemble-predictor.service --no-pager -n 20

echo ""
echo "✅ PyTorch installation complete!"
echo "🎯 Ensemble predictor should now produce real predictions (not fail-mode)"

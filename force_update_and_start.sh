#!/bin/bash
# Force update service.py from Windows and restart

echo "╔════════════════════════════════════════════╗"
echo "║   FORCE UPDATE & RESTART                   ║"
echo "╚════════════════════════════════════════════╝"
echo

echo "→ Copying updated service.py from Windows..."
cp /mnt/c/quantum_trader/microservices/ai_engine/service.py ~/quantum_trader/microservices/ai_engine/service.py
echo "✓ File copied"

echo "→ Verifying fix is present..."
if grep -q "NOTE: ServiceHealth removed" ~/quantum_trader/microservices/ai_engine/service.py; then
    echo "✓ Fix verified - ServiceHealth import corrected"
else
    echo "❌ WARNING: Fix not found in file!"
fi

echo "→ Cleaning Python cache..."
find ~/quantum_trader/microservices/ai_engine -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
find ~/quantum_trader/microservices/ai_engine -name "*.pyc" -delete 2>/dev/null
echo "✓ Cache cleaned"

echo "→ Stopping existing service..."
pkill -9 -f "uvicorn.*ai_engine" 2>/dev/null
sleep 1
echo "✓ Stopped"

echo "→ Starting service..."
cd ~/quantum_trader
source .venv/bin/activate
export PYTHONPATH=$HOME/quantum_trader
export REDIS_HOST=localhost
export REDIS_PORT=6379
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2

uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001

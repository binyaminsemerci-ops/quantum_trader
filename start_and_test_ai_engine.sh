#!/bin/bash
# Start AI Engine in detached mode and test health

cd ~/quantum_trader

# Kill existing
pkill -9 -f "uvicorn.*ai_engine" 2>/dev/null

# Copy updated service.py from Windows
cp /mnt/c/quantum_trader/microservices/ai_engine/service.py ~/quantum_trader/microservices/ai_engine/

# Clean cache
find ~/quantum_trader -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# Start service
source .venv/bin/activate
export PYTHONPATH=$HOME/quantum_trader
export REDIS_HOST=localhost
export REDIS_PORT=6379
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2

nohup uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001 > /tmp/ai_engine.log 2>&1 &
SERVICE_PID=$!

echo "Service started with PID: $SERVICE_PID"
sleep 5

# Test health
echo "Testing health endpoint..."
curl -s http://localhost:8001/health | python3 -m json.tool

# Show logs with error context
echo ""
echo "=== Recent logs (checking for errors) ==="
tail -30 /tmp/ai_engine.log | grep -E "Health check|create|ERROR|ServiceHealth" || tail -30 /tmp/ai_engine.log

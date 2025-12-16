#!/bin/bash
cd ~/quantum_trader
source .venv/bin/activate

# Clean environment: kill any accidental cross-path setup
unset PYTHONPATH
export PYTHONPATH="$HOME/quantum_trader"

echo "=== CLEAN ENVIRONMENT CHECK ==="
python - <<'PY'
import sys
bad = [p for p in sys.path if p.startswith("/mnt/c/") and "quantum_trader" in p.lower()]
print("BAD /mnt/c quantum_trader paths:", bad if bad else "NONE ✓")
PY

echo
echo "=== STOPPING OLD SERVICE ==="
pkill -f "uvicorn.*ai_engine.main" && echo "Stopped old service" || echo "No old service running"
sleep 2

echo
echo "=== STARTING AI ENGINE SERVICE ==="
echo "Starting uvicorn on port 8001..."
uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001 --log-level info

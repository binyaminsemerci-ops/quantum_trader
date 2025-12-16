#!/bin/bash
cd ~/quantum_trader

echo "🚀 Starting AI Engine service..."
source .venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$HOME/quantum_trader"

# Kill any old process
pkill -9 -f "uvicorn.*ai_engine.main" 2>/dev/null
sleep 2

# Start in background
nohup uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001 --log-level info > /tmp/ai_engine.log 2>&1 &
PID=$!
echo "Started with PID: $PID"
sleep 3

echo ""
echo "🔍 GREP ServiceHealth in service.py"
echo "─────────────────────────────────────────────────────────"
grep -n "ServiceHealth" microservices/ai_engine/service.py | head -50

echo ""
echo "🚨 CHECK FOR BAD IMPORT (should be EMPTY):"
if grep -q "from .models import.*ServiceHealth" microservices/ai_engine/service.py; then
    echo "✗✗✗ FEIL: 'from .models import ServiceHealth' FINNES FORTSATT!"
else
    echo "✓ GOOD: Ingen 'from .models import ServiceHealth' funnet"
fi

echo ""
echo "🧪 HARD VERIFY /health endpoint"
echo "─────────────────────────────────────────────────────────"
curl -sS http://localhost:8001/health | head -c 1000
echo ""
echo ""

RESPONSE=$(curl -sS http://localhost:8001/health)
if echo "$RESPONSE" | grep -q '"error".*"create"'; then
    echo "✗✗✗ FEIL: 'create' error found!"
else
    echo "✓ SUCCESS: NO 'create' error ✓"
fi

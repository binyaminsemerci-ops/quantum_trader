#!/bin/bash
echo "=== STOPPING ALL AI ENGINE PROCESSES ==="
pkill -9 -f "uvicorn.*ai_engine.main"
sleep 2

if pgrep -f "uvicorn.*ai_engine.main" > /dev/null; then
    echo "✗ Still running, trying harder..."
    pkill -9 -f "ai_engine"
    sleep 2
else
    echo "✓ All stopped"
fi

echo
echo "=== PORT CHECK ==="
if lsof -i :8001 2>/dev/null; then
    echo "⚠️  Something still on port 8001"
else
    echo "✓ Port 8001 is free"
fi

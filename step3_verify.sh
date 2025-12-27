#!/bin/bash
cd ~/quantum_trader
source .venv/bin/activate

echo "=== CLEAN PYCACHE (safe) ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "OK: cache cleaned"

echo
echo "=== TEST HEALTH ==="
RESPONSE=$(curl -sS http://localhost:8001/health)
echo "$RESPONSE" | head -c 1200
echo
echo

echo "=== ANALYSIS ==="
if echo "$RESPONSE" | grep -q '"error".*"create"'; then
    echo "✗✗✗ FEIL: 'create' error FORTSATT TIL STEDE!"
    echo "$RESPONSE" | grep -o '"error"[^,}]*'
else
    echo "✅ SUKSESS: Ingen 'create' error funnet!"
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status', 'N/A'))" 2>/dev/null)
    echo "Health status: $STATUS"
fi

echo
echo "=== QUICK LOG CHECK ==="
echo "Check running terminal for any 'Health check failed: create' messages"
echo "(Service is running in terminal ID: see background process)"

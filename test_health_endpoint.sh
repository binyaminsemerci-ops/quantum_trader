#!/bin/bash
cd ~/quantum_trader

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SERVICE HEALTH TEST                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Sjekk om service kjører
if pgrep -f "uvicorn.*ai_engine.main" > /dev/null; then
    echo "✓ AI Engine service kjører (PID: $(pgrep -f 'uvicorn.*ai_engine.main'))"
else
    echo "✗ AI Engine service kjører IKKE"
    echo "Start service først med:"
    echo "  wsl bash ~/quantum_trader/start_ai_engine_wsl.sh"
    exit 1
fi

echo ""
echo "🧪 Test /health endpoint"
echo "─────────────────────────────────────────────────────────"

RESPONSE=$(curl -s http://localhost:8001/health)
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

echo ""
echo "📊 ANALYSE"
echo "═════════════════════════════════════════════════════════"

if echo "$RESPONSE" | grep -q '"error".*"create"'; then
    echo "✗ FEIL: 'create' error fortsatt til stede!"
    exit 1
elif echo "$RESPONSE" | grep -q '"status"'; then
    echo "✓ Health endpoint returnerer gyldig status"
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', 'UNKNOWN'))" 2>/dev/null)
    echo "  Status: $STATUS"
    echo ""
    echo "✓ SUKSESS: Ingen 'create' error funnet!"
else
    echo "⚠️  Ukjent response format"
fi

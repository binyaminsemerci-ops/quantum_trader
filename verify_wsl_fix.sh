#!/bin/bash
cd ~/quantum_trader

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  VERIFY FIX IN WSL REPO                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "🔍 GREP ServiceHealth in service.py"
echo "─────────────────────────────────────────────────────────"
grep -n "ServiceHealth" microservices/ai_engine/service.py | head -50

echo ""
echo "🚨 CHECK FOR BAD IMPORT (should be EMPTY):"
echo "─────────────────────────────────────────────────────────"
if grep -q "from .models import.*ServiceHealth" microservices/ai_engine/service.py; then
    echo "✗✗✗ FEIL: 'from .models import ServiceHealth' FINNES FORTSATT!"
    grep -n "from .models import.*ServiceHealth" microservices/ai_engine/service.py
else
    echo "✓ GOOD: Ingen 'from .models import ServiceHealth' funnet"
fi

echo ""
echo "🧪 HARD VERIFY /health endpoint"
echo "─────────────────────────────────────────────────────────"
curl -sS http://localhost:8001/health | head -c 1000
echo ""
echo ""

echo "📊 ANALYSIS"
echo "─────────────────────────────────────────────────────────"
RESPONSE=$(curl -sS http://localhost:8001/health)
if echo "$RESPONSE" | grep -q '"error".*"create"'; then
    echo "✗✗✗ FEIL: 'create' error found in response!"
else
    echo "✓ SUCCESS: NO 'create' error in response"
fi

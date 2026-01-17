#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "P0 FIX PACK — PHASE 2: CONSUMER GROUPS + TERMINAL STATE PROOF"
echo "========================================================================"
echo ""

echo "[1/5] Checking consumer groups on trade.intent stream..."
redis-cli XINFO GROUPS quantum:stream:trade.intent 2>&1 | head -20 || echo "No groups yet"

if redis-cli XINFO GROUPS quantum:stream:trade.intent 2>&1 | grep -q "quantum:group:execution"; then
    echo "✅ Consumer group exists!"
    
    echo ""
    echo "[2/5] Checking consumers..."
    redis-cli XINFO CONSUMERS quantum:stream:trade.intent quantum:group:execution:trade.intent 2>&1 | head -20 || echo "No consumers"
    
    echo ""
    echo "[3/5] Checking pending entries..."
    redis-cli XPENDING quantum:stream:trade.intent quantum:group:execution:trade.intent - + 20 2>&1 || echo "No pending"
    
else
    echo "⚠️  No consumer group found yet (may need more time or new events)"
fi

echo ""
echo "[4/5] Checking execution logs for consumer group info..."
journalctl -u quantum-execution.service -n 100 --no-pager | grep -E "Consumer group|Consumer name|subscribe" | tail -10

echo ""
echo "[5/5] Checking for terminal state logs..."
echo "Looking for TERMINAL STATE entries..."
journalctl -u quantum-execution.service -n 100 --no-pager | grep "TERMINAL STATE" | tail -10 || echo "No terminal states yet (waiting for new intents)"

echo ""
echo "========================================================================"
echo "WAITING FOR NEW TRADE INTENT..."
echo "========================================================================"
echo ""

echo "Monitoring execution logs (30s)..."
echo "(Looking for: Consumer group creation, TERMINAL STATE logs)"
echo ""

timeout 30 journalctl -u quantum-execution.service -f --no-pager | grep --line-buffered -E "Consumer|TERMINAL|TradeIntent received|FILLED|REJECTED" || true

echo ""
echo "========================================================================"
echo "PHASE 2 PROOF COMPLETE"
echo "========================================================================"
echo ""

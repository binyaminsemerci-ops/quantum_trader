#!/bin/bash
# M3 investigation script
echo "=== PHASE timeout locations ==="
grep -rn "Funding rate timeout\|Orderbook timeout\|Volatility timeout\|Risk prediction timeout\|PHASE.*timeout\|timeout.*PHASE" /home/qt/quantum_trader/ai_engine/ 2>/dev/null | grep -v ".pyc\|#" | head -20

echo ""
echo "=== Files with 5.0 timeout in ai_engine ==="
grep -rn "timeout=5[^0-9]\|asyncio.wait_for.*timeout=5" /home/qt/quantum_trader/ai_engine/ 2>/dev/null | grep -v ".pyc\|#\|twitter\|coingecko" | head -20

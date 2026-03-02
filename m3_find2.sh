#!/bin/bash
# M3 fix: Find timeout source and increase from 5s to 10s

echo "=== Finding Funding rate timeout in /opt/quantum ==="
grep -rn "Funding rate timeout\|Orderbook timeout\|Volatility timeout\|Risk prediction timeout" /opt/quantum/ai_engine/ 2>/dev/null | grep -v ".pyc" | head -20

echo ""
echo "=== Files with 5.0 timeout in /opt/quantum/ai_engine ==="
grep -rn "timeout=5[^0-9]\|asyncio.wait_for.*timeout=5" /opt/quantum/ai_engine/ 2>/dev/null | grep -v ".pyc\|#\|twitter\|coingecko\|backup\|emoji" | head -30

#!/bin/bash
echo "=== Search for dynamic timeout log construction ==="
grep -rn "PHASE.*timeout\|timeout.*PHASE\|Funding rate\|funding.*5.*s\|Orderbook.*5\|Volatility.*5\|Risk prediction.*5" /opt/quantum/ai_engine/ 2>/dev/null | grep -v ".pyc\|emoji" | head -30

echo ""
echo "=== Search in backend for these log patterns ==="
grep -rn "Funding rate timeout\|Orderbook timeout\|Volatility timeout\|Risk prediction timeout" /opt/quantum/backend/ 2>/dev/null | grep -v ".pyc\|emoji" | head -20

echo ""
echo "=== Which service is producing logs ==="
# Check the ai_engine main script
ls /opt/quantum/ai_engine/

#!/bin/bash
# M3 fix: Find timeout source and increase from 5s to 10s

echo "=== Lines 1070-1130 of xgb_agent.py ==="
sed -n '1070,1140p' /home/qt/quantum_trader/ai_engine/agents/xgb_agent.py

echo ""
echo "=== Checking if Funding rate timeout log text exists ==="
grep -n "Funding rate timeout\|Orderbook timeout\|Volatility timeout\|Risk prediction timeout" /home/qt/quantum_trader/ai_engine/agents/xgb_agent.py 2>/dev/null | head -20

echo ""
echo "=== Check all ai_engine .py files for phase timeout messages ==="
grep -rn "Funding rate timeout\|Orderbook timeout\|Volatility timeout\|Risk prediction timeout" /home/qt/quantum_trader/ai_engine/ 2>/dev/null | grep -v ".pyc" | head -20

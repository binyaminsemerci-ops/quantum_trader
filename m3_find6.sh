#!/bin/bash
echo "=== PHASE timeout in microservices ==="
grep -rn "Funding rate timeout\|Orderbook timeout\|Volatility timeout\|Risk prediction timeout\|\[PHASE 1\].*timeout\|\[PHASE 2" /opt/quantum/microservices/ 2>/dev/null | grep -v ".pyc" | head -30

echo ""
echo "=== ai_engine service main ==="
find /opt/quantum/microservices/ -name "*.py" -path "*/ai_engine/*" 2>/dev/null | head -10
ls /opt/quantum/microservices/ 2>/dev/null

echo ""  
echo "=== systemd service for ai_engine ==="
grep "ExecStart\|WorkingDir" /etc/systemd/system/quantum-ai-engine.service 2>/dev/null

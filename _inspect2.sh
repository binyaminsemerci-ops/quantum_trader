#!/bin/bash
echo "=== APPLY_LAYER CONFIG (lines 757-870) ==="
sed -n '757,870p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== APPLY_LAYER MAIN LOOP / DECISION (lines 1290-1450) ==="
sed -n '1290,1450p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== APPLY_LAYER OPEN LOGIC (lines 1450-1600) ==="
sed -n '1450,1600p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== INTENT_EXECUTOR COOLDOWN / RATE LIMIT (lines 390-470) ==="
sed -n '390,470p' /opt/quantum/microservices/intent_executor/main.py

echo ""
echo "=== APPLY_LAYER SYMBOL SCANNING (lines 1600-1700) ==="
sed -n '1600,1700p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== RUNNING SERVICE ENV VARS ==="
for svc in quantum-apply-layer quantum-intent-executor quantum-harvest-brain quantum-harvest-v2; do
    echo "--- $svc ---"
    systemctl show $svc --property=Environment 2>/dev/null | tr ' ' '\n' | grep -v "^Environment=" | head -20
    echo ""
done

echo ""
echo "=== APPLY_LAYER SCORE OPEN condition (lines 1700-1900) ==="
sed -n '1700,1900p' /opt/quantum/microservices/apply_layer/main.py

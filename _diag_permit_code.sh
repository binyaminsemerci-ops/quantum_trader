#!/bin/bash
echo "=== home/qt intent_executor lines 385-420 (_wait_for_permit) ==="
sed -n '385,425p' /home/qt/quantum_trader/microservices/intent_executor/main.py

echo ""
echo "=== home/qt intent_executor lines 975-1010 (inline permit check) ==="
sed -n '975,1015p' /home/qt/quantum_trader/microservices/intent_executor/main.py

echo ""
echo "=== import json in home/qt intent_executor? ==="
grep -n "^import json\|^from json\|import json" /home/qt/quantum_trader/microservices/intent_executor/main.py | head -5

echo ""
echo "=== position_state_brain: full permit setex lines ==="
sed -n '790,840p' /opt/quantum/microservices/position_state_brain/main.py

echo ""
echo "=== P3.3 config.PERMIT_TTL value ==="
grep -n "PERMIT_TTL" /opt/quantum/microservices/position_state_brain/main.py | head -5

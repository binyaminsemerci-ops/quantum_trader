#!/bin/bash
echo "=== APPLY_LAYER LINES 1368-1415 (step builder PARTIAL section) ==="
sed -n '1368,1415p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== GATE 5 CLOSE LIST (lines 1312-1322) ==="
sed -n '1312,1327p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== GATE 6 CLOSE LIST (lines 1335-1341) ==="
sed -n '1334,1342p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== NORMALIZE_ACTION PASS-THROUGH LINE ==="
grep -n "FULL_CLOSE_PROPOSED.*PARTIAL_75.*PARTIAL_50" /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== CURRENT EXCHANGE BRIDGE ENV ==="
cat /etc/quantum/exchange-stream-bridge.env

echo ""
echo "=== BRIDGE redis cfg key ==="
redis-cli EXISTS quantum:cfg:universe:active
redis-cli TYPE quantum:cfg:universe:active 2>/dev/null

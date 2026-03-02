#!/bin/bash
echo "=== APPLY_LAYER LINES 1395-1430 ==="
sed -n '1395,1430p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== APPLY_LAYER normalize_action FUNCTION ==="
grep -n "def normalize_action\|normalize_action" /opt/quantum/microservices/apply_layer/main.py | head -20

echo ""
echo "=== APPLY_LAYER normalize_action BODY ==="
# Find the line number of normalize_action def and print 60 lines from there
LINE=$(grep -n "def normalize_action" /opt/quantum/microservices/apply_layer/main.py | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    sed -n "${LINE},$((LINE+80))p" /opt/quantum/microservices/apply_layer/main.py
fi

echo ""
echo "=== UNIVERSE ACTIVE ALL SYMBOLS ==="
redis-cli SMEMBERS quantum:universe:active 2>/dev/null | sort

echo ""
echo "=== EXCHANGE BRIDGE SOURCE FILE ==="
ls -la /home/qt/quantum_trader/microservices/data_collector/exchange_stream_bridge.py 2>/dev/null
grep -n "USE_UNIVERSE\|universe\|active\|SMEMBERS\|symbol_list\|symbols" /home/qt/quantum_trader/microservices/data_collector/exchange_stream_bridge.py 2>/dev/null | head -30

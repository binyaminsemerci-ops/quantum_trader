#!/bin/bash
echo "=== UNIVERSE ACTIVE KEY TYPE ==="
redis-cli TYPE quantum:universe:active

echo ""
echo "=== UNIVERSE ACTIVE CONTENT ==="
redis-cli GET quantum:universe:active 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys())[:5] if isinstance(d,dict) else str(d)[:500])" 2>/dev/null || redis-cli GET quantum:universe:active 2>/dev/null | head -c 500

echo ""
echo "=== UNIVERSE ACTIVE (LRANGE if list) ==="
redis-cli LRANGE quantum:universe:active 0 10 2>/dev/null | head -20

echo ""
echo "=== UNIVERSE ACTIVE (SMEMBERS if set) ==="
redis-cli SMEMBERS quantum:universe:active 2>/dev/null | head -20

echo ""
echo "=== EXCHANGE.RAW LAST ENTRY ==="
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 1 2>/dev/null

echo ""
echo "=== APPLY_LAYER LINES 1330-1370 ==="
sed -n '1330,1370p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== APPLY_LAYER LINES 1240-1260 (first SKIP) ==="
sed -n '1240,1265p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== APPLY_LAYER LINES 1295-1320 (second SKIP) ==="
sed -n '1295,1320p' /opt/quantum/microservices/apply_layer/main.py

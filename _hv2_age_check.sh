#!/bin/bash
echo "=== quantum:config:harvest_v2 ==="
redis-cli hgetall quantum:config:harvest_v2

echo ""
echo "=== PositionProvider instantiation in harvest_v2.py ==="
grep -n "PositionProvider\|max_age\|pos_provider\|HARVEST_V2" /opt/quantum/microservices/harvest_v2/harvest_v2.py | head -20

echo ""
echo "=== env check ==="
systemctl show quantum-harvest-v2 | grep -i "HARVEST"

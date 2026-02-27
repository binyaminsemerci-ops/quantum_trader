#!/bin/bash
echo "=== WHO WRITES TO harvest.v2.shadow ==="
grep -r "harvest.v2.shadow" /opt/quantum/microservices/ /home/qt/quantum_trader/ 2>/dev/null \
  | grep -v Binary | grep -v node_modules | grep -v ".pyc" | head -30

echo ""
echo "=== WHO WRITES TO trade.closed ==="
grep -r "trade.closed" /opt/quantum/microservices/ /home/qt/quantum_trader/ 2>/dev/null \
  | grep -v Binary | grep -v node_modules | grep -v ".pyc" | head -30

echo ""
echo "=== HARVEST V2 RELATED SERVICES ==="
systemctl list-units --type=service --all | grep -i "harvest"

echo ""
echo "=== CONSUMER GROUPS ON harvest.v2.shadow ==="
redis-cli xinfo groups quantum:stream:harvest.v2.shadow

echo ""
echo "=== LAST 3 MESSAGES harvest.v2.shadow ==="
redis-cli xrevrange quantum:stream:harvest.v2.shadow + - COUNT 3

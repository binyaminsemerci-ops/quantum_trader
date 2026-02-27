#!/bin/bash
echo "=== SANDBOX / TESTNET SERVICES ==="
systemctl list-units --type=service --all | grep -i sandbox || echo "none"

echo ""
echo "=== TESTNET ENV ==="
grep -r "TESTNET\|sandbox" /opt/quantum/.env /etc/quantum/*.env 2>/dev/null | head -20

echo ""
echo "=== SANDBOX RELATED PROCESSES ==="
ps aux | grep -i sandbox | grep -v grep || echo "none"

echo ""
echo "=== SANDBOX FILES ==="
find /home/qt/quantum_trader -name "*sandbox*" -o -name "*testnet*" 2>/dev/null | head -20

echo ""
echo "=== DOCKER SANDBOX ==="
docker ps -a --format "{{.Names}} | {{.Status}} | {{.Image}}" 2>/dev/null | grep -i sandbox || echo "no sandbox containers"

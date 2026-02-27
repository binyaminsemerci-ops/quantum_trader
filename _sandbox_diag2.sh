#!/bin/bash
echo "=== harvest-v2 LAST 20 LOG LINES ==="
journalctl -u quantum-harvest-v2 -n 20 --no-pager

echo ""
echo "=== harvest-v2 SERVICE FILE ==="
cat /etc/systemd/system/quantum-harvest-v2.service

echo ""
echo "=== harvest-v2 ENV FILE ==="
cat /etc/quantum/harvest-v2.env 2>/dev/null || echo "no harvest-v2.env"
cat /etc/quantum/harvest_v2.env 2>/dev/null || echo "no harvest_v2.env"

echo ""
echo "=== trade.closed LAST MSG ==="
redis-cli xrevrange quantum:stream:trade.closed + - COUNT 2

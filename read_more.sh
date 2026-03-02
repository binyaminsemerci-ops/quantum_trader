#!/bin/bash
echo "=== auto_permit_p33.py context ==="
sed -n '28,50p' /opt/quantum/scripts/auto_permit_p33.py

echo ""
echo "=== harvest_brain home dir line 920-935 ==="
sed -n '920,938p' /home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py

echo ""
echo "=== Is auto_permit_p33 running? ==="
ps aux | grep auto_permit | grep -v grep || echo "not running as a process"
crontab -l 2>/dev/null | grep auto_permit || echo "not in crontab"

echo ""
echo "=== quantum:metrics key sample ==="
redis-cli KEYS "quantum:metrics:*" 2>/dev/null | head -5
redis-cli TTL "$(redis-cli KEYS 'quantum:metrics:*' 2>/dev/null | head -1)" 2>/dev/null

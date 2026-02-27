#!/bin/bash
echo "=== Governor service status ==="
systemctl is-active quantum-governor

echo ""
echo "=== Governor recent log ==="
journalctl -u quantum-governor --no-pager -n 20 -q 2>&1 | tail -20

echo ""
echo "=== Sample permit keys in Redis ==="
redis-cli keys "quantum:permit:governor:*" 2>/dev/null | head -5
redis-cli keys "quantum:permit:p33:*" 2>/dev/null | head -5

echo ""
echo "=== Anti-churn keys after recent SELLs ==="
redis-cli keys "quantum:intent_bridge:last_close:*" 2>/dev/null

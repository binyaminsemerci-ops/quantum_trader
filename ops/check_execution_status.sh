#!/bin/bash
echo "=== apply_layer recent activity ==="
journalctl -u quantum-apply-layer --no-pager --since "2 minutes ago" -q 2>/dev/null | tail -50

echo ""
echo "=== executed=True in apply.result? ==="
redis-cli xrevrange quantum:stream:apply.result + - COUNT 50 2>/dev/null | grep -c "True"

echo ""
echo "=== AAVEUSDT position current state ==="
redis-cli hgetall quantum:state:positions:AAVEUSDT 2>/dev/null

echo ""
echo "=== ACEUSDT position current state ==="
redis-cli hgetall quantum:state:positions:ACEUSDT 2>/dev/null

echo ""
echo "=== Active position count (canonical keys) ==="
redis-cli keys "quantum:state:positions:*" 2>/dev/null | wc -l

echo ""
echo "=== Last 5 apply.result entries ==="
redis-cli xrevrange quantum:stream:apply.result + - COUNT 5 2>/dev/null

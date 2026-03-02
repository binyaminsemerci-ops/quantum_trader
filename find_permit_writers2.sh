#!/bin/bash
echo "=== Files that set quantum:permit ==="
grep -rn "\.set.*permit\|\.hset.*permit\|\.setex.*permit\|setnx.*permit\|permit_key.*=\|gov_key.*set\|p33_key.*set" /opt/quantum/microservices/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak\|apply_layer\|#\|consume\|get\|exists\|delete" | head -25

echo ""
echo "=== Governor service permit writes ==="
find /opt/quantum/microservices/ /opt/quantum/backend/ -name "governor*.py" -o -name "position_state*.py" 2>/dev/null | head -10

echo ""
echo "=== Try direct set pattern with permit keys ==="
grep -rn "r\.set.*gov_key\|r\.set.*p33_key\|redis\.set.*permit\|\.set(f.quantum:permit" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji\|.bak" | head -20

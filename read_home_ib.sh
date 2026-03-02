#!/bin/bash
echo "=== Read intent_bridge home version around permit_key hset ==="
grep -n "permit_key\|hset\|expire" /home/qt/quantum_trader/microservices/intent_bridge/main.py | grep -E "permit|hset|expire" | tail -20

echo ""
echo "=== Full context around the hset permit write ==="
LINENO=$(grep -n "auto_bypass_no_p33\|auto-bypass\|auto_bypass" /home/qt/quantum_trader/microservices/intent_bridge/main.py | tail -1 | cut -d: -f1)
if [ -n "$LINENO" ]; then
    echo "Target near line: $LINENO"
    START=$((LINENO - 8))
    END=$((LINENO + 6))
    sed -n "${START},${END}p" /home/qt/quantum_trader/microservices/intent_bridge/main.py
fi

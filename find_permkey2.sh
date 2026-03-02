#!/bin/bash
echo "=== intent_executor permit_key usage ==="
grep -n "permit_key\|hset\|setex\|expire" /opt/quantum/microservices/intent_executor/main.py | grep -A2 -B2 "permit" | head -40

echo ""
echo "=== All hset calls with permit_key across microservices ==="
grep -rn "hset.*permit_key\|hset(permit" /opt/quantum/microservices/ 2>/dev/null | grep -v "H1 fix"

echo ""
echo "=== Any setex with permit_key ==="
grep -rn "setex.*permit_key\|setex(permit" /opt/quantum/microservices/ 2>/dev/null

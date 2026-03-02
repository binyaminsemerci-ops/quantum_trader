#!/bin/bash
echo "=== Governor permit creation ==="
grep -rn "quantum:permit" /opt/quantum/microservices/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak\|apply_layer\|#" | grep "\.py:" | grep -v "get\|consume\|exists\|delete\|ttl\|check\|comment" | head -20

echo ""
echo "=== Governor main.py permit writes ==="
find /opt/quantum/microservices/ -name "*.py" 2>/dev/null | xargs grep -l "quantum:permit" 2>/dev/null | grep -v "apply_layer\|portfolio_gate" | head -10

echo ""
echo "=== portfolio_gate permit creation ==="
grep -n "set\|expire\|permit" /opt/quantum/microservices/portfolio_gate/main.py 2>/dev/null | grep -i "permit" | head -20

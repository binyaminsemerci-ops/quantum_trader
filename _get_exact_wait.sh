#!/bin/bash
echo "=== _wait_for_permit exact lines ==="
grep -n "_wait_for_permit\|hgetall" /home/qt/quantum_trader/microservices/intent_executor/main.py | head -20

echo ""
echo "=== Lines 393-430 ==="
sed -n '393,435p' /home/qt/quantum_trader/microservices/intent_executor/main.py | cat -A | head -50

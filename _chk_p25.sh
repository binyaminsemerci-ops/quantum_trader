#!/bin/bash
echo "=== elif action == PARTIAL_25 check ==="
grep -n 'elif action == "PARTIAL_25"' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== Lines 1390-1415 ==="
sed -n '1390,1415p' /opt/quantum/microservices/apply_layer/main.py

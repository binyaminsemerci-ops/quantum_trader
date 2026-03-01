#!/bin/bash
echo "=== INTENT_EXECUTOR: source allowlist check logic (lines 790-870) ==="
sed -n '785,870p' /opt/quantum/microservices/intent_executor/main.py

echo ""
echo "=== INTENT_EXECUTOR: allowlist definition ==="
grep -n "allowlist\|ALLOWLIST\|source_allow\|HARVEST_V2\|harvest_v2" /opt/quantum/microservices/intent_executor/main.py | head -30

echo ""
echo "=== INTENT_EXECUTOR: what streams does it read? ==="
grep -n "apply.plan\|XREAD\|stream\b" /opt/quantum/microservices/intent_executor/main.py | head -20

echo ""
echo "=== INTENT_EXECUTOR: SKIP result write ==="
sed -n '800,860p' /opt/quantum/microservices/intent_executor/main.py

echo ""
echo "=== apply.result: check who is actually writing SKIP ==="
# check if apply_layer writes to apply.result
grep -n "apply.result\|apply_result" /opt/quantum/microservices/apply_layer/main.py | head -20

echo ""
echo "=== INTENT_EXECUTOR: apply.result write ==="
grep -n "apply.result\|apply_result" /opt/quantum/microservices/intent_executor/main.py | head -20

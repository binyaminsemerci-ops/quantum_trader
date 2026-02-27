#!/bin/bash
echo "=== GOVERNOR BYPASS LOCATION ==="
grep -rn "Governor bypass" /opt/quantum/microservices/ 2>/dev/null | head -10
grep -rn "TESTNET MODE ENABLED" /opt/quantum/microservices/ 2>/dev/null | head -5

echo "=== EXITBRAIN STREAM OUTPUT ==="
grep -n "xadd\|apply.plan\|trade.intent" /opt/quantum/microservices/exitbrain_v3_5/service.py 2>/dev/null | head -25

echo "=== APPLY_LAYER LIB DIR ==="
ls -la /opt/quantum/microservices/apply_layer/lib/ 2>/dev/null

echo "=== APPLY_LAYER GOVERNOR LINES 1840-1870 ==="
sed -n '1840,1880p' /opt/quantum/microservices/apply_layer/main.py 2>/dev/null

echo "=== INTENT_BRIDGE IMPORTS ==="
head -30 /opt/quantum/microservices/intent_bridge/main.py 2>/dev/null

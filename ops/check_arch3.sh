#!/bin/bash
echo "=== GOVERNOR BYPASS BLOCK (lines 95-145) ==="
sed -n '95,145p' /opt/quantum/microservices/apply_layer/main.py

echo "=== APPLY_LAYER TESTNET LINES ==="
grep -n "TESTNET\|testnet\|bypass\|governor" /opt/quantum/microservices/apply_layer/main.py | head -30

echo "=== INTENT_BRIDGE COST_MODEL IMPORT ==="
grep -n "cost_model\|CostModel\|breakeven" /opt/quantum/microservices/intent_bridge/main.py | head -10

echo "=== APPLY_LAYER LIB DIR EXISTENCE ==="
ls -la /opt/quantum/microservices/apply_layer/lib/ 2>/dev/null || echo "lib/ dir does not exist"

echo "=== INTENT_BRIDGE PROCESS_INTENT TAIL (before publish) ==="
grep -n "publish_plan\|exposure\|_publish" /opt/quantum/microservices/intent_bridge/main.py | tail -20

echo "=== EXITBRAIN FULL STREAM REFS ==="
grep -n "stream\|xadd\|PLAN\|intent" /opt/quantum/microservices/exitbrain_v3_5/service.py 2>/dev/null | head -30

#!/bin/bash
echo "=== EXIT_OWNERSHIP USAGE IN APPLY_LAYER ==="
grep -n "exit_ownership\|EXIT_OWNER\|validate_exit" /opt/quantum/microservices/apply_layer/main.py | head -30

echo "=== APPLY_LAYER LINES 35-65 ==="
sed -n '35,65p' /opt/quantum/microservices/apply_layer/main.py

echo "=== HOW exit_ownership IS CALLED (search for function calls) ==="
grep -n "validate_exit_ownership\|EXIT_OWNER" /opt/quantum/microservices/apply_layer/main.py

echo "=== APPLY_LAYER APPLYMODE ENUM ==="
grep -n "ApplyMode\|apply_mode\|TESTNET\|mode =" /opt/quantum/microservices/apply_layer/main.py | grep -v "^#" | head -30

echo "=== APPLY_LAYER INIT SECTION (lines 800-870) ==="
sed -n '800,870p' /opt/quantum/microservices/apply_layer/main.py

echo "=== EXITBRAIN service.py first 100 lines ==="
head -100 /opt/quantum/microservices/exitbrain_v3_5/service.py 2>/dev/null

echo "=== ALL SERVICES WRITING TO apply.plan ==="
grep -rn "apply.plan\|apply_plan" /opt/quantum/microservices/*/service.py 2>/dev/null | grep "xadd\|publish\|PLAN_STREAM" | head -20

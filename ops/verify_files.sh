#!/bin/bash
echo "=== VERIFY intent_bridge deployment ==="
grep -n "ANTI_CHURN\|MIN_HOLD\|BUILD_TAG\|ROUND_TRIP" /opt/quantum/microservices/intent_bridge/main.py | head -20

echo ""
echo "=== VERIFY apply_layer governor fix ==="
grep -n "GOVERNOR_BYPASS" /opt/quantum/microservices/apply_layer/main.py | head -10

echo ""
echo "=== VERIFY exit_ownership ==="
ls -la /opt/quantum/microservices/apply_layer/lib/
python3 -c "import sys; sys.path.insert(0, '/opt/quantum/microservices/apply_layer'); from lib.exit_ownership import EXIT_OWNER, validate_exit_ownership; print(f'EXIT_OWNER={EXIT_OWNER} OK')"

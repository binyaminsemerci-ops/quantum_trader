#!/bin/bash
echo "=== Clearing Python caches ==="
find /opt/quantum/microservices/intent_bridge -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
find /opt/quantum/microservices/apply_layer -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "Cache cleared"

echo "=== Restarting services ==="
systemctl restart quantum-intent-bridge quantum-apply-layer
sleep 8

echo "=== intent_bridge startup (BUILD_TAG check) ==="
journalctl -u quantum-intent-bridge --no-pager -n 30 -q 2>&1 | grep -E "BUILD_TAG|Intent Bridge|anti.churn|started"

echo ""
echo "=== apply_layer startup (governor check) ==="
journalctl -u quantum-apply-layer --no-pager -n 30 -q 2>&1 | grep -E "Governor active|GOVERNOR_BYPASS|TESTNET execution|EXIT_OWNER|exit_own"

echo ""
echo "=== SERVICES ACTIVE ==="
systemctl is-active quantum-intent-bridge
systemctl is-active quantum-apply-layer

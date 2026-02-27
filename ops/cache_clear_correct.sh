#!/bin/bash
QT="/home/qt/quantum_trader"

echo "=== Clearing Python caches (correct path) ==="
find $QT/microservices/intent_bridge -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
find $QT/microservices/apply_layer -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "Cache cleared"

echo "=== Restarting services ==="
systemctl restart quantum-intent-bridge quantum-apply-layer
sleep 8

echo "=== intent_bridge BUILD_TAG ==="
journalctl -u quantum-intent-bridge --no-pager -n 20 -q 2>&1 | grep "Intent Bridge"

echo ""
echo "=== apply_layer governor status ==="
journalctl -u quantum-apply-layer --no-pager -n 20 -q 2>&1 | grep -E "Governor|GOVERNOR|EXIT_OWNER|exit_own|TESTNET"

echo ""
echo "=== SERVICES ACTIVE ==="
systemctl is-active quantum-intent-bridge
systemctl is-active quantum-apply-layer

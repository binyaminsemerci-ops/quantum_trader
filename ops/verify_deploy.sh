#!/bin/bash
echo "=== intent_bridge startup log ==="
journalctl -u quantum-intent-bridge --no-pager -n 30 | grep -E "BUILD_TAG|anti.churn|ANTI_CHURN|Intent Bridge =|MIN_HOLD|ROUND_TRIP"

echo ""
echo "=== apply_layer startup log ==="
journalctl -u quantum-apply-layer --no-pager -n 30 | grep -E "Governor active|GOVERNOR_BYPASS|TESTNET execution|exit_own|EXIT_OWNER|EXIT_OWNERSHIP"

echo ""
echo "=== exit_ownership import check ==="
journalctl -u quantum-apply-layer --no-pager -n 50 | grep -iE "exit_own|WARN: exit_own"

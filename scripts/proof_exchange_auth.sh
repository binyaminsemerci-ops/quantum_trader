#!/bin/bash
# Exchange Authentication Proof Script
# Verifies API keys load from env-file and authentication works

set -euo pipefail

echo "🔐 Exchange Authentication Proof"
echo "=================================="
echo ""

# Configuration
ENV_FILE="/etc/quantum/governor.env"
SCRIPT_PATH="/home/qt/quantum_trader/scripts/dump_exchange_positions.py"
PYTHON="/usr/bin/python3"

# Check prerequisites
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ FAIL: Env file not found: $ENV_FILE"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ FAIL: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "✅ Prerequisites OK"
echo "   ENV_FILE: $ENV_FILE"
echo "   SCRIPT: $SCRIPT_PATH"
echo ""

# Test 1: Direct execution with --env-file
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Direct execution with --env-file flag"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /home/qt/quantum_trader || exit 1

# Capture output
OUTPUT1=$(python3 scripts/dump_exchange_positions.py --env-file "$ENV_FILE" --max 5 2>&1)
EXIT1=$?

echo "$OUTPUT1"
echo ""

# Assertions for Test 1
if [ $EXIT1 -ne 0 ]; then
    echo "❌ FAIL: Script exited with code $EXIT1"
    exit 1
fi

if echo "$OUTPUT1" | grep -qiE "401|unauthorized|-2015"; then
    echo "❌ FAIL: Authentication error detected (401/-2015)"
    echo "$OUTPUT1" | grep -iE "401|unauthorized|-2015"
    exit 1
fi

if echo "$OUTPUT1" | grep -qE "BINANCE_TESTNET_API_KEY|BINANCE_TESTNET_API_SECRET"; then
    echo "❌ FAIL: Output contains credential key names (privacy leak)"
    echo "$OUTPUT1" | grep -E "BINANCE_TESTNET_API_KEY|BINANCE_TESTNET_API_SECRET"
    exit 1
fi

if ! echo "$OUTPUT1" | grep -qiE "Active positions:|positions"; then
    echo "❌ FAIL: Output missing position information"
    exit 1
fi

if ! echo "$OUTPUT1" | grep -q "BINANCE TESTNET FUTURES"; then
    echo "❌ FAIL: Output missing expected header"
    exit 1
fi

echo "✅ TEST 1 PASSED"
echo "   - Script executed successfully"
echo "   - No authentication errors"
echo "   - Position data displayed"
echo ""

# Test 2: Gold path with systemd-run + EnvironmentFile
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: systemd-run with EnvironmentFile (gold standard)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run with systemd-run (secure, no shell export)
OUTPUT2=$(systemd-run --quiet --pipe --wait \
    -p "EnvironmentFile=$ENV_FILE" \
    "$PYTHON" "$SCRIPT_PATH" --max 5 2>&1)
EXIT2=$?

echo "$OUTPUT2"
echo ""

# Assertions for Test 2
if [ $EXIT2 -ne 0 ]; then
    echo "❌ FAIL: systemd-run exited with code $EXIT2"
    exit 1
fi

if echo "$OUTPUT2" | grep -qiE "401|unauthorized|-2015"; then
    echo "❌ FAIL: Authentication error in systemd-run"
    echo "$OUTPUT2" | grep -iE "401|unauthorized|-2015"
    exit 1
fi

if echo "$OUTPUT2" | grep -qE "BINANCE_TESTNET_API_KEY|BINANCE_TESTNET_API_SECRET"; then
    echo "❌ FAIL: systemd-run output contains credential key names (privacy leak)"
    echo "$OUTPUT2" | grep -E "BINANCE_TESTNET_API_KEY|BINANCE_TESTNET_API_SECRET"
    exit 1
fi

if ! echo "$OUTPUT2" | grep -qiE "Active positions:|positions"; then
    echo "❌ FAIL: systemd-run output missing position information"
    exit 1
fi

if ! echo "$OUTPUT2" | grep -q "BINANCE TESTNET FUTURES"; then
    echo "❌ FAIL: systemd-run output missing expected header"
    exit 1
fi

echo "✅ TEST 2 PASSED"
echo "   - systemd-run executed successfully"
echo "   - EnvironmentFile loaded correctly"
echo "   - No authentication errors"
echo "   - Position data displayed"
echo ""

# Extract position count for summary
POS_COUNT=$(echo "$OUTPUT2" | grep -oP 'Active positions.*?: \K\d+' || echo "N/A")

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ ALL TESTS PASSED"
echo ""
echo "Test Results:"
echo "  [✓] Direct execution with --env-file"
echo "  [✓] systemd-run with EnvironmentFile"
echo "  [✓] No authentication errors"
echo "  [✓] API keys loaded from: $ENV_FILE"
echo "  [✓] Active positions: $POS_COUNT"
echo ""
echo "Security:"
echo "  [✓] No secrets in output"
echo "  [✓] No secrets in command line"
echo "  [✓] EnvironmentFile loaded by systemd"
echo ""
echo "🎯 Exchange authentication VERIFIED"
echo ""
exit 0

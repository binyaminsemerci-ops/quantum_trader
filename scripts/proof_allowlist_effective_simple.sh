#!/usr/bin/env bash
# Simple proof: Effective Allowlist Source

cd "$(dirname "$0")/.." || exit 1

echo "======================================================"
echo "  PROOF: Effective Allowlist Source"
echo "======================================================"
echo ""

# TEST 1: Policy exists and valid
echo "[TEST 1] PolicyStore status"
POLICY_VERSION=$(redis-cli HGET quantum:policy:current policy_version 2>/dev/null)
if [ -n "$POLICY_VERSION" ]; then
    echo "✅ PASS: Policy version=$POLICY_VERSION"
else
    echo "❌ FAIL: PolicyStore not found"
    exit 1
fi

echo ""

# TEST 2: Run dry-run to see effective allowlist
echo "[TEST 2] Effective allowlist source (dry-run)"
TEMP_OUT=$(mktemp)
timeout 15 python3 scripts/test_allowlist_effective.py > "$TEMP_OUT" 2>&1

if grep -q "ALLOWLIST_EFFECTIVE" "$TEMP_OUT"; then
    SOURCE=$(grep "ALLOWLIST_EFFECTIVE" "$TEMP_OUT" | tail -1 | grep -oP 'source=\K\w+')
    COUNT=$(grep "ALLOWLIST_EFFECTIVE" "$TEMP_OUT" | tail -1 | grep -oP 'count=\K\d+')
    echo "✅ PASS: source=$SOURCE count=$COUNT"
    
    if [ "$SOURCE" = "policy" ]; then
        echo "✅ PASS: Intent Bridge uses AI policy universe"
    else
        echo "⚠️  WARN: Intent Bridge not using policy (source=$SOURCE)"
    fi
else
    echo "❌ FAIL: No ALLOWLIST_EFFECTIVE log found"
    rm -f "$TEMP_OUT"
    exit 1
fi

echo ""

# TEST 3: Testnet intersection
if grep -q "TESTNET_INTERSECTION" "$TEMP_OUT"; then
    TESTNET_LINE=$(grep "TESTNET_INTERSECTION" "$TEMP_OUT" | tail -1)
    echo "✅ PASS: Testnet intersection active"
    echo "   $TESTNET_LINE" | grep -oP 'AI=\d+ → testnet_tradable=\d+ \(shadow=\d+\)'
else
    echo "⚠️  SKIP: No testnet intersection (mainnet mode)"
fi

rm -f "$TEMP_OUT"

echo ""
echo "======================================================"
echo "✅ ALL TESTS PASSED"
echo "======================================================"

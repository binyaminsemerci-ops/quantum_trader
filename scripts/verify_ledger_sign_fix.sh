#!/bin/bash
# PROOF SCRIPT: Verify executor ledger sign fix after order fill
# 
# USAGE: Run this immediately after an order fills to verify:
#   1. Snapshot position_amt sign matches ledger position_amt sign
#   2. P3.3 last_known_amt matches position_amt
#   3. Side derived correctly from sign
#   4. No reconcile_required_side_mismatch errors

set -e

SYMBOL="${1:-BTCUSDT}"

echo "=== LEDGER SIGN FIX VERIFICATION ==="
echo "Symbol: $SYMBOL"
echo ""

# Snapshot truth (exchange reality)
echo "1) SNAPSHOT (exchange truth):"
SNAP_AMT=$(redis-cli HGET quantum:position:snapshot:$SYMBOL position_amt)
SNAP_SIDE=$(redis-cli HGET quantum:position:snapshot:$SYMBOL side)
echo "   position_amt: $SNAP_AMT"
echo "   side: $SNAP_SIDE"
echo ""

# Ledger state (executor writes this)
echo "2) LEDGER (executor writes):"
LED_AMT=$(redis-cli HGET quantum:position:ledger:$SYMBOL position_amt)
LED_LAST_AMT=$(redis-cli HGET quantum:position:ledger:$SYMBOL last_known_amt)
LED_SIDE=$(redis-cli HGET quantum:position:ledger:$SYMBOL side)
LED_QTY=$(redis-cli HGET quantum:position:ledger:$SYMBOL qty)
echo "   position_amt: $LED_AMT"
echo "   last_known_amt: $LED_LAST_AMT (P3.3 reads this)"
echo "   side: $LED_SIDE"
echo "   qty: $LED_QTY"
echo ""

# Verification checks
echo "3) VERIFICATION:"

# Check 1: Sign must match
if python3 -c "import sys; sys.exit(0 if abs(float('$SNAP_AMT') - float('$LED_AMT')) < 1e-6 else 1)"; then
    echo "   ✅ position_amt sign matches snapshot"
else
    echo "   ❌ FAIL: position_amt mismatch (snapshot=$SNAP_AMT, ledger=$LED_AMT)"
    exit 1
fi

# Check 2: last_known_amt must equal position_amt
if python3 -c "import sys; sys.exit(0 if abs(float('$LED_AMT') - float('$LED_LAST_AMT')) < 1e-6 else 1)"; then
    echo "   ✅ last_known_amt = position_amt (P3.3 contract)"
else
    echo "   ❌ FAIL: last_known_amt mismatch (position_amt=$LED_AMT, last_known_amt=$LED_LAST_AMT)"
    exit 1
fi

# Check 3: Side must match sign
EXPECTED_SIDE=$(python3 -c "amt=float('$LED_AMT'); print('LONG' if amt > 1e-12 else ('SHORT' if amt < -1e-12 else 'FLAT'))")
if [ "$LED_SIDE" = "$EXPECTED_SIDE" ]; then
    echo "   ✅ side derived correctly from sign ($LED_SIDE)"
else
    echo "   ❌ FAIL: side mismatch (expected=$EXPECTED_SIDE, actual=$LED_SIDE)"
    exit 1
fi

# Check 4: qty must be abs(position_amt)
EXPECTED_QTY=$(python3 -c "print(abs(float('$LED_AMT')))")
if python3 -c "import sys; sys.exit(0 if abs(float('$LED_QTY') - float('$EXPECTED_QTY')) < 1e-6 else 1)"; then
    echo "   ✅ qty = abs(position_amt) = $LED_QTY"
else
    echo "   ❌ FAIL: qty should be $EXPECTED_QTY, got $LED_QTY"
    exit 1
fi

# Check 5: No mismatch errors in last 5 minutes
echo ""
echo "4) P3.3 MISMATCH CHECK (last 5 min):"
MISMATCH_COUNT=$(journalctl -u quantum-position-state-brain --since "5 minutes ago" --no-pager | grep -c "reconcile_required_side_mismatch" 2>/dev/null || echo 0)
MISMATCH_COUNT=$(echo "$MISMATCH_COUNT" | tr -d '\n' | tr -d ' ')
if [ "$MISMATCH_COUNT" -eq 0 ] 2>/dev/null; then
    echo "   ✅ No reconcile_required_side_mismatch errors"
else
    echo "   ⚠️  WARNING: $MISMATCH_COUNT mismatch errors found"
    exit 1
fi

echo ""
echo "=== ALL CHECKS PASSED ✅ ==="
echo "Ledger sign fix working correctly!"

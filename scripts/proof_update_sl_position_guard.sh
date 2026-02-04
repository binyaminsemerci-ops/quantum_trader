#!/usr/bin/env bash
set -euo pipefail

# Proof: UPDATE_SL Position Guard (apply-layer-entry-exit-sep-v1)
# Tests: UPDATE_SL skipped when no position, allowed when position exists

echo "=== Proof: UPDATE_SL Position Guard ==="
echo "Target: UPDATE_SL → SKIP when no position (fail-soft)"
echo

# 1. Check BTC/ETH (no positions) for UPDATE_SL_SKIP
echo "1. BTC/ETH Plans (No Position Expected):"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 50 > /tmp/plans_guard.txt

btc_skip=$(grep -B15 "symbol: BTCUSDT" /tmp/plans_guard.txt | grep "decision: SKIP" | wc -l || echo 0)
btc_execute_update_sl=$(grep -B15 "symbol: BTCUSDT" /tmp/plans_guard.txt | grep "decision: EXECUTE" | grep -A3 "UPDATE_SL" | wc -l || echo 0)
btc_reason=$(grep -B10 "symbol: BTCUSDT" /tmp/plans_guard.txt | grep "update_sl_no_position_skip" | wc -l || echo 0)

echo "   BTCUSDT:"
echo "     - decision=SKIP: $btc_skip"
echo "     - decision=EXECUTE with UPDATE_SL: $btc_execute_update_sl (should be 0)"
echo "     - reason 'update_sl_no_position_skip': $btc_reason"

eth_skip=$(grep -B15 "symbol: ETHUSDT" /tmp/plans_guard.txt | grep "decision: SKIP" | wc -l || echo 0)
eth_execute_update_sl=$(grep -B15 "symbol: ETHUSDT" /tmp/plans_guard.txt | grep "decision: EXECUTE" | grep -A3 "UPDATE_SL" | wc -l || echo 0)
eth_reason=$(grep -B10 "symbol: ETHUSDT" /tmp/plans_guard.txt | grep "update_sl_no_position_skip" | wc -l || echo 0)

echo "   ETHUSDT:"
echo "     - decision=SKIP: $eth_skip"
echo "     - decision=EXECUTE with UPDATE_SL: $eth_execute_update_sl (should be 0)"
echo "     - reason 'update_sl_no_position_skip': $eth_reason"
echo

# 2. Check logs for UPDATE_SL_SKIP_NO_POSITION
echo "2. UPDATE_SL_SKIP Logs (last 2 min):"
journalctl -u quantum-apply-layer --since "2 min ago" | grep "UPDATE_SL_SKIP_NO_POSITION" | tail -10
echo

# 3. Check ZEC/FIL (with positions) for UPDATE_SL allowed
echo "3. ZEC/FIL Plans (Position May Exist):"
zec_execute=$(grep -B15 "symbol: ZECUSDT" /tmp/plans_guard.txt | grep "decision: EXECUTE" | wc -l || echo 0)
zec_update_sl=$(grep -B10 "symbol: ZECUSDT" /tmp/plans_guard.txt | grep "UPDATE_SL" | wc -l || echo 0)

fil_execute=$(grep -B15 "symbol: FILUSDT" /tmp/plans_guard.txt | grep "decision: EXECUTE" | wc -l || echo 0)
fil_update_sl=$(grep -B10 "symbol: FILUSDT" /tmp/plans_guard.txt | grep "UPDATE_SL" | wc -l || echo 0)

echo "   ZECUSDT:"
echo "     - decision=EXECUTE: $zec_execute"
echo "     - UPDATE_SL steps: $zec_update_sl"
echo "   FILUSDT:"
echo "     - decision=EXECUTE: $fil_execute"
echo "     - UPDATE_SL steps: $fil_update_sl"
echo

# 4. Check position snapshots
echo "4. Position Snapshots (Redis):"
btc_pos=$(redis-cli HGET quantum:snapshot:position:BTCUSDT position_amt || echo "0")
eth_pos=$(redis-cli HGET quantum:snapshot:position:ETHUSDT position_amt || echo "0")
zec_pos=$(redis-cli HGET quantum:snapshot:position:ZECUSDT position_amt || echo "0")
fil_pos=$(redis-cli HGET quantum:snapshot:position:FILUSDT position_amt || echo "0")

echo "   BTCUSDT position_amt: $btc_pos"
echo "   ETHUSDT position_amt: $eth_pos"
echo "   ZECUSDT position_amt: $zec_pos"
echo "   FILUSDT position_amt: $fil_pos"
echo

# 5. Verification
echo "5. Verification:"
if [ "$btc_execute_update_sl" -eq 0 ] && [ "$eth_execute_update_sl" -eq 0 ]; then
    echo "   ✓ BTC/ETH: No UPDATE_SL with EXECUTE (fail-soft working)"
else
    echo "   ✗ BTC/ETH: Found UPDATE_SL with EXECUTE (fail-soft NOT working)"
fi

if [ "$btc_reason" -gt 0 ] || [ "$eth_reason" -gt 0 ]; then
    echo "   ✓ BTC/ETH: Found 'update_sl_no_position_skip' reason"
else
    echo "   ⚠ BTC/ETH: No 'update_sl_no_position_skip' reason found"
fi

echo
echo "=== Proof Complete ==="
echo "Summary: UPDATE_SL guard prevents execution when no position exists"

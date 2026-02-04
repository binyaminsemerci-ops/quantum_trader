#!/usr/bin/env bash
set -euo pipefail

# Proof: Apply Layer Entry/Exit Separation (apply-layer-entry-exit-sep-v1)
# Tests: BUILD_TAG, config, OPEN/CLOSE decisions, BTC/ETH EXECUTE rate, qty_scale

echo "=== Proof: Apply Layer Entry/Exit Separation ==="
echo "Target: BTC/ETH EXECUTE rate >20% (was 0%)"
echo

# 1. BUILD_TAG verification
echo "1. BUILD_TAG Check:"
if journalctl -u quantum-apply-layer --since '5 min ago' | grep -q "apply-layer-entry-exit-sep-v1"; then
    echo "   ✓ BUILD_TAG found"
else
    echo "   ✗ BUILD_TAG NOT found (check deploy)"
    exit 1
fi
echo

# 2. Config verification
echo "2. Config Check:"
journalctl -u quantum-apply-layer --since '5 min ago' | grep "Entry/Exit:" | tail -1
echo

# 3. Decision logs (OPEN vs CLOSE)
echo "3. Decision Logs (last 50 lines):"
journalctl -u quantum-apply-layer --since '2 min ago' | grep -E "OPEN|CLOSE" | tail -20
echo

# 4. qty_scale logs
echo "4. Qty Scale Logs:"
journalctl -u quantum-apply-layer --since '2 min ago' | grep "qty_scale" | tail -10
echo

# 5. BTC/ETH execute rate
echo "5. BTC/ETH Execute Rate:"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 200 > /tmp/plans_check.txt

btc_total=$(grep -c "symbol: BTCUSDT" /tmp/plans_check.txt || true)
btc_exec=$(grep -B5 "symbol: BTCUSDT" /tmp/plans_check.txt | grep "decision: EXECUTE" | wc -l || true)
btc_blocked=$(grep -B5 "symbol: BTCUSDT" /tmp/plans_check.txt | grep "decision: BLOCKED" | wc -l || true)

eth_total=$(grep -c "symbol: ETHUSDT" /tmp/plans_check.txt || true)
eth_exec=$(grep -B5 "symbol: ETHUSDT" /tmp/plans_check.txt | grep "decision: EXECUTE" | wc -l || true)
eth_blocked=$(grep -B5 "symbol: ETHUSDT" /tmp/plans_check.txt | grep "decision: BLOCKED" | wc -l || true)

echo "   BTCUSDT: $btc_exec EXECUTE / $btc_blocked BLOCKED (total: $btc_total)"
echo "   ETHUSDT: $eth_exec EXECUTE / $eth_blocked BLOCKED (total: $eth_total)"

# Calculate rates
if [ "$btc_total" -gt 0 ]; then
    btc_rate=$(awk "BEGIN {printf \"%.1f\", ($btc_exec/$btc_total)*100}")
    echo "   BTC Execute Rate: $btc_rate%"
    
    if [ $(echo "$btc_rate > 20" | bc -l) -eq 1 ]; then
        echo "   ✓ BTC rate >20% (SUCCESS)"
    else
        echo "   ⚠ BTC rate <20% (needs tuning)"
    fi
fi

if [ "$eth_total" -gt 0 ]; then
    eth_rate=$(awk "BEGIN {printf \"%.1f\", ($eth_exec/$eth_total)*100}")
    echo "   ETH Execute Rate: $eth_rate%"
fi
echo

# 6. Reason code distribution
echo "6. Reason Code Distribution:"
echo "   Old codes:"
grep -o "kill_score_warning_risk_increase" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_warning_risk_increase:"
grep -o "kill_score_warning_close_ok" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_warning_close_ok:"
echo "   New codes:"
grep -o "kill_score_open_scaled" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_open_scaled:"
grep -o "kill_score_open_ok" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_open_ok:"
grep -o "kill_score_open_critical" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_open_critical:"
grep -o "kill_score_close_blocked" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_close_blocked:"
grep -o "kill_score_close_ok" /tmp/plans_check.txt | wc -l | xargs echo "     kill_score_close_ok:"
echo

echo "=== Proof Complete ==="

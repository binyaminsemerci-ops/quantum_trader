#!/bin/bash
# P0.5 MarketState Metrics Publisher — Verification Script
# Read-only checks to verify metrics are being published

set -e

echo "================================================================================"
echo "P0.5 MarketState Metrics Publisher — Verification"
echo "================================================================================"
echo ""

# Check 1: Service status
echo "CHECK 1: Service status"
echo "--------------------------------------------------------------------------------"
systemctl is-active quantum-marketstate.service && echo "✅ Service is active" || echo "❌ Service is not active"
systemctl is-enabled quantum-marketstate.service && echo "✅ Service is enabled" || echo "⚠️  Service is not enabled"
echo ""

# Check 2: Recent logs
echo "CHECK 2: Recent logs (last 20 lines)"
echo "--------------------------------------------------------------------------------"
journalctl -u quantum-marketstate -n 20 --no-pager
echo ""

# Check 3: Redis metrics exist
echo "CHECK 3: Redis metrics (checking BTCUSDT)"
echo "--------------------------------------------------------------------------------"
redis-cli HGETALL quantum:marketstate:BTCUSDT | head -20
echo ""

# Check 4: Stream entries
echo "CHECK 4: Recent stream entries (last 5)"
echo "--------------------------------------------------------------------------------"
redis-cli XREVRANGE quantum:stream:marketstate + - COUNT 5
echo ""

# Check 5: All tracked symbols
echo "CHECK 5: All tracked symbols"
echo "--------------------------------------------------------------------------------"
redis-cli KEYS 'quantum:marketstate:*' | while read key; do
    echo "Key: $key"
    ts=$(redis-cli HGET "$key" ts)
    regime_trend=$(redis-cli HGET "$key" p_trend)
    regime_mr=$(redis-cli HGET "$key" p_mr)
    regime_chop=$(redis-cli HGET "$key" p_chop)
    timestamp=$(redis-cli HGET "$key" ts_timestamp)
    
    if [ -n "$ts" ]; then
        age=$(($(date +%s) - timestamp))
        echo "  TS=$ts | Trend=$regime_trend MR=$regime_mr Chop=$regime_chop | Age=${age}s"
    else
        echo "  ⚠️  No data"
    fi
    echo ""
done

echo "================================================================================"
echo "Verification complete!"
echo "================================================================================"
echo ""
echo "To monitor live:"
echo "  journalctl -u quantum-marketstate -f"
echo ""
echo "To check specific symbol:"
echo "  redis-cli HGETALL quantum:marketstate:BTCUSDT"
echo ""
echo "To restart service:"
echo "  systemctl restart quantum-marketstate.service"
echo ""

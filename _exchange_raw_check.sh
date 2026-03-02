#!/bin/bash
echo "=== exchange.raw stream check ==="
redis-cli XLEN quantum:stream:exchange.raw 2>/dev/null
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 1 2>/dev/null | head -20

echo ""
echo "=== quantum:meta:regime current value ==="
redis-cli GET quantum:meta:regime
redis-cli TTL quantum:meta:regime

echo ""
echo "=== Latest signals: risk_context changed? ==="
redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 5 2>/dev/null | head -60

echo ""
echo "=== Intent executor: any new executions? ==="
journalctl -u quantum-intent-executor.service -n 15 --no-pager 2>/dev/null | grep -E "execut|✅|🚀|⏳|ERROR|SKIP"

echo ""
echo "=== Apply.result: last executed=True entry ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 200 2>/dev/null | grep -B3 "executed" | grep -A1 "^executed$" | grep "True" | head -5

echo ""
echo "=== What is quantum:stream:exchange.raw? Which service writes it? ==="
grep -r "exchange.raw\|exchange_raw" /opt/quantum --include="*.py" -l 2>/dev/null | head -10

echo ""
echo "=== POPULATE exchange.raw from market.klines ==="
# The meta-regime needs exchange.raw with price data
# We can bridge the existing market.klines data to exchange.raw format
echo "Checking if cross-exchange service exists..."
systemctl list-units 'quantum-cross-exchange*' --no-pager 2>/dev/null
systemctl list-units 'quantum-*exchange*' --no-pager 2>/dev/null | head -10

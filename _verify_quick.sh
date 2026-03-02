#!/bin/bash
sleep 8
echo "=== INTENT EXECUTOR (last 10) ==="
journalctl -u quantum-intent-executor.service -n 10 --no-pager

echo ""
echo "=== META-REGIME (last 8) ==="
journalctl -u quantum-meta-regime.service -n 8 --no-pager

echo ""
echo "=== Regime key ==="
redis-cli GET quantum:meta:regime
redis-cli KEYS 'quantum:regime*'

echo ""
echo "=== BNB position ==="
redis-cli HGETALL quantum:position:BNBUSDT

echo ""
echo "=== Latest executed trades ==="
redis-cli XREVRANGE quantum:stream:trade.execution.res + - COUNT 5 2>/dev/null | head -40

echo ""
echo "=== apply.result last 5 ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 2>/dev/null | grep -E "plan_id|symbol|executed|error" | head -30

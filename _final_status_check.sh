#!/bin/bash
echo "=== quantum:cfg:universe:meta type (WRONGTYPE candidate) ==="
redis-cli TYPE quantum:cfg:universe:meta

echo ""
echo "=== New apply.plan entries — any OPEN plans? ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 5 2>/dev/null

echo ""
echo "=== Intent executor: executions after restart ==="
journalctl -u quantum-intent-executor.service -n 20 --no-pager 2>/dev/null | grep -E "execut|🚀|SKIP|GUARD|WRONG" | tail -15

echo ""
echo "=== Current 4 positions with close viability ==="
for sym in ADAUSDT SOLUSDT BTCUSDT ETHUSDT; do
    side=$(redis-cli HGET quantum:position:$sym side 2>/dev/null)
    qty=$(redis-cli HGET quantum:position:$sym quantity 2>/dev/null)
    entry=$(redis-cli HGET quantum:position:$sym entry_price 2>/dev/null)
    echo "  $sym: $side qty=$qty entry=$entry"
done

echo ""
echo "=== Total PnL check (quick) ==="
redis-cli KEYS 'quantum:position:snapshot:BTCUSDT' 2>/dev/null
redis-cli HGET quantum:position:snapshot:BTCUSDT unrealized_pnl 2>/dev/null
redis-cli HGET quantum:position:snapshot:ETHUSDT unrealized_pnl 2>/dev/null
redis-cli HGET quantum:position:snapshot:SOLUSDT unrealized_pnl 2>/dev/null
redis-cli HGET quantum:position:snapshot:ADAUSDT unrealized_pnl 2>/dev/null

echo ""
echo "=== Executed true count now ==="
redis-cli HGET quantum:metrics:intent_executor executed_true 2>/dev/null
journalctl -u quantum-intent-executor.service -n 5 --no-pager 2>/dev/null | grep "executed_true"

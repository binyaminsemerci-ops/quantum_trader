#!/bin/bash
echo "=== What was plan 982ceeb1? ==="
redis-cli HGETALL "quantum:result:982ceeb1" 2>/dev/null
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 20 2>/dev/null | grep -B5 "982ceeb1" | head -20

echo ""
echo "=== Latest apply.result (raw, few entries) ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3 2>/dev/null

echo ""
echo "=== BNB position state now ==="
redis-cli HGETALL quantum:position:BNBUSDT
redis-cli HGETALL quantum:position:snapshot:BNBUSDT

echo ""
echo "=== p34 position bootstrap logs ==="
journalctl -u quantum-p34-position-bootstrap.service -n 10 --no-pager 2>/dev/null || \
journalctl -u quantum-p3*bootstrap* -n 10 --no-pager 2>/dev/null || \
systemctl list-units 'quantum-p3*service' --no-pager 2>/dev/null | head -15

echo ""
echo "=== Meta-regime source code: what stream does it read? ==="
grep -n "stream\|kline\|market\|XREAD\|xread\|XREVRANGE\|COUNT" \
    /home/qt/quantum_trader/microservices/meta_regime/main.py 2>/dev/null | head -25

echo ""
echo "=== P3.5 guard: why is layer4_kelly_200usdt blocking close? ==="
grep -n "kelly_200usdt\|kelly.*200\|200usdt\|close.*skip\|CLOSE.*kelly" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -20

echo ""
echo "=== Current open positions in Redis (all key types) ==="
echo "--- quantum:position:* (position objects) ---"
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | while read k; do
    side=$(redis-cli HGET "$k" side 2>/dev/null)
    qty=$(redis-cli HGET "$k" quantity 2>/dev/null)
    sym=$(redis-cli HGET "$k" symbol 2>/dev/null)
    src=$(redis-cli HGET "$k" source 2>/dev/null)
    echo "  $k: $side qty=$qty src=$src"
done

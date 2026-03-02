#!/bin/bash
echo "=== RESTART apply_layer to reload in-memory position cache ==="
systemctl restart quantum-apply-layer.service 2>/dev/null
sleep 4
echo "Status: $(systemctl is-active quantum-apply-layer.service 2>/dev/null)"
journalctl -u quantum-apply-layer.service -n 8 --no-pager 2>/dev/null | tail -10

echo ""
echo "=== FIND WRONGTYPE source: check potential keys ==="
# Check the seen_orders key type (used with sadd)
redis-cli KEYS 'quantum:*seen*' 2>/dev/null | head -10
redis-cli KEYS 'quantum:*orders*seen*' 2>/dev/null | head -10
redis-cli KEYS 'quantum:executor:*' 2>/dev/null | head -10
for key in $(redis-cli KEYS 'quantum:executor:*' 2>/dev/null | head -20); do
    ktype=$(redis-cli TYPE "$key" 2>/dev/null)
    echo "  $key => $ktype"
done

echo ""
echo "=== Check for string-type keys that should be hash/set ==="
redis-cli KEYS 'quantum:intent*' 2>/dev/null | while read k; do
    ktype=$(redis-cli TYPE "$k" 2>/dev/null)
    echo "  $k => $ktype"
done | head -20

echo ""
echo "=== Check apply_layer new plans after restart (wait 8s) ==="
sleep 8
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 2>/dev/null | grep -E "plan_id|symbol|action|close_qty|decision|source" | head -30

echo ""
echo "=== Check apply.result after restart ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 2>/dev/null | grep -E "plan_id|symbol|executed|error" | head -25

echo ""
echo "=== Intent executor WRONGTYPE still occurring? ==="
journalctl -u quantum-intent-executor.service -n 10 --no-pager 2>/dev/null | grep -E "WRONGTYPE|execut" | tail -8

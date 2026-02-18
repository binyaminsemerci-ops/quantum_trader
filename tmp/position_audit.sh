#!/bin/bash
# Position audit script - investigate all quantum:position:* keys

echo "=== POSITION AUDIT $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""

echo "--- Total position keys ---"
redis-cli KEYS "quantum:position:*" | wc -l

echo ""
echo "--- Non-zero positions (qty != 0) ---"
count=0
for key in $(redis-cli KEYS "quantum:position:*"); do
    qty=$(redis-cli HGET "$key" quantity)
    if [ -n "$qty" ] && [ "$qty" != "0" ] && [ "$qty" != "0.0" ]; then
        ((count++))
    fi
done
echo "$count positions with non-zero quantity"

echo ""
echo "--- Sample of 10 positions (all fields) ---"
redis-cli KEYS "quantum:position:*" | head -10 | while read key; do
    symbol=${key#quantum:position:}
    echo "Symbol: $symbol"
    redis-cli HGETALL "$key"
    echo "---"
done

echo ""
echo "--- Position summary by side ---"
redis-cli KEYS "quantum:position:*" | while read key; do
    redis-cli HGET "$key" side
done | sort | uniq -c

echo ""
echo "--- Checking for ghost positions (qty=0 but key exists) ---"
ghost_count=0
for key in $(redis-cli KEYS "quantum:position:*"); do
    qty=$(redis-cli HGET "$key" quantity 2>/dev/null)
    if [ "$qty" = "0" ] || [ "$qty" = "0.0" ] || [ -z "$qty" ]; then
        ((ghost_count++))
    fi
done
echo "$ghost_count ghost positions (qty=0 or missing)"

echo ""
echo "--- Binance actual positions (from account) ---"
# This will require API call - placeholder for now
echo "(Requires Binance API call - will implement next)"

echo ""
echo "=== END AUDIT ==="

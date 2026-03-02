#!/bin/bash
# Check if permit keys have TTL or are permanent (-1)
echo "Sampling 5 permit key TTLs:"
redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -5 | while read key; do
    ttl=$(redis-cli TTL "$key")
    echo "  $key → TTL=$ttl"
done

echo ""
echo "Counting permanent keys (TTL=-1):"
redis-cli KEYS "quantum:permit:*" 2>/dev/null | while read key; do
    redis-cli TTL "$key"
done | grep -c "^-1" || echo "0 permanent keys"

echo "DONE"

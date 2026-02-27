#!/bin/bash
echo "=== PORTFOLIO KEYS ==="
redis-cli keys "quantum:portfolio:*" | head -20
echo "=== RISK KEYS ==="
redis-cli keys "quantum:risk:*" | head -20
echo "=== INTENT_EXECUTOR KEYS ==="
redis-cli keys "quantum:intent_executor:*" | head -10
echo "=== STATS KEYS ==="
redis-cli keys "quantum:stats:*" | head -10
echo "=== HARVEST ENV ==="
cat /etc/quantum/harvest-proposal.env
echo "=== ENTRY=0 POSITIONS ==="
for key in $(redis-cli keys "quantum:position:*" | grep -v snapshot | grep -v backup); do
  entry=$(redis-cli hget "$key" entry_price 2>/dev/null)
  if [ "$entry" = "0.0" ] || [ "$entry" = "0" ]; then
    sym=$(redis-cli hget "$key" symbol 2>/dev/null)
    echo "  PHANTOM entry=0: $key ($sym)"
  fi
done

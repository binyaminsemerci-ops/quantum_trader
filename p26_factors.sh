#!/bin/bash

echo "=== P2.6: K factor fields (if present) ==="
for s in BTCUSDT ETHUSDT SOLUSDT; do
  echo "--- $s ---"
  redis-cli HGETALL quantum:harvest:proposal:$s | grep -E 'kill|K|regime|sigma|ts|age|component|raw' || true
  echo
done

#!/bin/bash
echo "=== PROOF 3: K-score sanity check ==="
for s in ETHUSDT SOLUSDT; do
  echo "--- $s ---"
  redis-cli HMGET quantum:harvest:proposal:$s kill_score k_regime_flip k_ts_drop
  echo
done

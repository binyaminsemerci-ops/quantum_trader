#!/bin/bash
echo "=== PROOF 1: k_components fields exist in Redis ==="
for s in BTCUSDT ETHUSDT SOLUSDT; do
  echo "--- $s ---"
  redis-cli HMGET quantum:harvest:proposal:$s k_regime_flip k_sigma_spike k_ts_drop k_age_penalty
  echo
done

#!/bin/bash

echo "=== P2.6 SNAPSHOT (inputs/outputs) ==="
date -u
echo

for s in BTCUSDT ETHUSDT SOLUSDT; do
  echo "--- $s ---"
  echo "[marketstate]"
  redis-cli HGETALL quantum:marketstate:$s | head -60
  echo "[risk proposal]"
  redis-cli HGETALL quantum:risk:proposal:$s | head -60
  echo "[harvest proposal]"
  redis-cli HGETALL quantum:harvest:proposal:$s | head -80
  echo
done

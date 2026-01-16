#!/bin/bash

echo "=== CHECKING CONTEXT & REGIME KEYS ==="
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'
redis-cli KEYS "market:*"
echo "---REGIME---"
redis-cli KEYS "*regime*"
echo "---UNIVERSE---"
redis-cli KEYS "universe:*"
echo "---CONTEXT---"
redis-cli KEYS "*context*"
ENDSSH

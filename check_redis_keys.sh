#!/bin/bash

echo "=== CHECKING CONTEXT & REGIME KEYS ==="
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'
docker exec quantum_redis redis-cli KEYS "market:*"
echo "---REGIME---"
docker exec quantum_redis redis-cli KEYS "*regime*"
echo "---UNIVERSE---"
docker exec quantum_redis redis-cli KEYS "universe:*"
echo "---CONTEXT---"
docker exec quantum_redis redis-cli KEYS "*context*"
ENDSSH

#!/bin/bash
set -e

VPS="root@46.224.116.254"
KEY="~/.ssh/hetzner_fresh"
QT="/opt/quantum"

echo "=== Step 1: Create lib/ directory ==="
ssh -i $KEY $VPS "mkdir -p $QT/microservices/apply_layer/lib"

echo "=== Step 2: Deploy exit_ownership ==="
scp -i $KEY /mnt/c/quantum_trader/microservices/apply_layer/lib/__init__.py $VPS:$QT/microservices/apply_layer/lib/__init__.py
scp -i $KEY /mnt/c/quantum_trader/microservices/apply_layer/lib/exit_ownership.py $VPS:$QT/microservices/apply_layer/lib/exit_ownership.py

echo "=== Step 3: Deploy intent_bridge ==="
scp -i $KEY /mnt/c/quantum_trader/microservices/intent_bridge/main.py $VPS:$QT/microservices/intent_bridge/main.py

echo "=== Step 4: Deploy apply_layer ==="
scp -i $KEY /mnt/c/quantum_trader/microservices/apply_layer/main.py $VPS:$QT/microservices/apply_layer/main.py

echo "=== Step 5: Restart services (docker) ==="
ssh -i $KEY $VPS "
  echo 'Restarting intent_bridge...'
  docker restart quantum_intent_bridge 2>/dev/null || docker restart intent_bridge 2>/dev/null || echo 'intent_bridge container name unknown, checking...'
  docker ps --format '{{.Names}}' | grep -i intent

  echo 'Restarting apply_layer...'
  docker restart quantum_apply_layer 2>/dev/null || docker restart apply_layer 2>/dev/null || echo 'apply_layer container name unknown, checking...'
  docker ps --format '{{.Names}}' | grep -i apply
"

echo "=== Step 6: Verify deployments ==="
sleep 5
ssh -i $KEY $VPS "
  echo '--- intent_bridge build tag ---'
  docker logs --tail 15 \$(docker ps --format '{{.Names}}' | grep -i intent | head -1) 2>/dev/null | grep -E 'BUILD_TAG|anti.churn|ANTI_CHURN|Intent Bridge'

  echo '--- apply_layer governor status ---'
  docker logs --tail 15 \$(docker ps --format '{{.Names}}' | grep -i apply | head -1) 2>/dev/null | grep -E 'Governor|GOVERNOR|TESTNET execution'

  echo '--- exit_ownership status ---'
  docker logs --tail 20 \$(docker ps --format '{{.Names}}' | grep -i apply | head -1) 2>/dev/null | grep -E 'exit_own|EXIT_OWNER|EXIT_OWNERSHIP'
"

echo "=== DONE ==="

#!/bin/bash
stream="quantum:stream:trade.intent"
group="quantum:group:quantum_trader:trade.intent"

echo "=== CLEANUP PENDING MESSAGES ==="
for i in {1..20}; do
  pending=$(docker exec quantum_redis redis-cli XPENDING $stream $group - + 50)
  if [ -z "$pending" ]; then 
    echo "No more pending messages"
    break
  fi
  
  echo "Batch $i: Processing 50 pending messages..."
  
  ids=$(echo "$pending" | awk '{print $1}' | head -50)
  for id in $ids; do
    docker exec quantum_redis redis-cli XACK $stream $group $id > /dev/null
  done
done

echo ""
echo "=== FINAL STATUS ==="
docker exec quantum_redis redis-cli XINFO GROUPS $stream

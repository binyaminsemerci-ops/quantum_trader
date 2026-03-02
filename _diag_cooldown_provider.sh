#!/bin/bash
echo "=== intent_executor: cooldown/last_exec_ts lines ==="
grep -n "last_exec_ts\|cooldown" /opt/quantum/microservices/intent_executor/main.py | head -40

echo ""
echo "=== intent_executor: SET with no TTL? ==="
grep -n "redis.*set\|r\.set\|\.set(" /opt/quantum/microservices/intent_executor/main.py | grep -iv "hset\|hmset\|smembers\|setex\|pipeline\|#" | head -20

echo ""
echo "=== position_provider.py FULL ==="
cat /home/qt/quantum_trader/microservices/harvest_v2/feeds/position_provider.py

echo ""
echo "=== reconcile_engine: snapshot key creation ==="
grep -n "snapshot\|position:snapshot" /opt/quantum/microservices/reconcile_engine/main.py | head -30

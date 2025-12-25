#!/bin/bash

echo "=== CHECKING IF LIVE TRADING OR PAPER MODE ==="
echo ""

ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'

echo "1. CHECK TRADING MODE IN CEO BRAIN:"
docker logs quantum_ceo_brain --tail 50 | grep -i "trade\|position\|order\|paper\|live\|mode" | tail -20
echo ""

echo "2. CHECK FOR ACTIVE POSITIONS:"
docker exec quantum_redis redis-cli KEYS "*position*"
echo ""

echo "3. CHECK FOR RECENT TRADES:"
docker exec quantum_redis redis-cli KEYS "*trade*"
echo ""

echo "4. CHECK FOR ORDERS:"
docker exec quantum_redis redis-cli KEYS "*order*"
echo ""

echo "5. CHECK EXECUTION ENGINE LOGS:"
docker logs quantum_execution_engine --tail 30 | grep -i "trade\|order\|fill" | tail -15
echo ""

echo "6. CHECK STRATEGY BRAIN RECENT ACTIVITY:"
docker logs quantum_strategy_brain --tail 30 | grep -i "signal\|trade\|position" | tail -15
echo ""

echo "7. CHECK AI ENGINE TRADING STATUS:"
docker logs quantum_ai_engine --tail 30 | grep -i "trade\|position\|paper" | tail -15
echo ""

echo "8. CHECK REDIS FOR TRADING CONFIG:"
docker exec quantum_redis redis-cli GET quantum:config:trading_mode
docker exec quantum_redis redis-cli GET quantum:config:paper_trading
docker exec quantum_redis redis-cli GET trading_mode
docker exec quantum_redis redis-cli GET paper_trading
echo ""

echo "9. CHECK FOR BALANCE/CAPITAL:"
docker exec quantum_redis redis-cli KEYS "*balance*"
docker exec quantum_redis redis-cli KEYS "*capital*"
docker exec quantum_redis redis-cli KEYS "*equity*"
echo ""

echo "10. CHECK LAST EXECUTION METRICS:"
docker exec quantum_redis redis-cli GET execution_metrics

ENDSSH

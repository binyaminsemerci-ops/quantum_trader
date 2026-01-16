#!/bin/bash

echo "=== CHECKING IF LIVE TRADING OR PAPER MODE ==="
echo ""

ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'

echo "1. CHECK TRADING MODE IN CEO BRAIN:"
journalctl -u quantum-ceo-brain.service -n 50 --no-pager | grep -i "trade\|position\|order\|paper\|live\|mode" | tail -20
echo ""

echo "2. CHECK FOR ACTIVE POSITIONS:"
redis-cli KEYS "*position*"
echo ""

echo "3. CHECK FOR RECENT TRADES:"
redis-cli KEYS "*trade*"
echo ""

echo "4. CHECK FOR ORDERS:"
redis-cli KEYS "*order*"
echo ""

echo "5. CHECK EXECUTION ENGINE LOGS:"
journalctl -u quantum-execution-engine.service -n 30 --no-pager | grep -i "trade\|order\|fill" | tail -15
echo ""

echo "6. CHECK STRATEGY BRAIN RECENT ACTIVITY:"
journalctl -u quantum-strategy-brain.service -n 30 --no-pager | grep -i "signal\|trade\|position" | tail -15
echo ""

echo "7. CHECK AI ENGINE TRADING STATUS:"
journalctl -u quantum-ai-engine.service -n 30 --no-pager | grep -i "trade\|position\|paper" | tail -15
echo ""

echo "8. CHECK REDIS FOR TRADING CONFIG:"
redis-cli GET quantum:config:trading_mode
redis-cli GET quantum:config:paper_trading
redis-cli GET trading_mode
redis-cli GET paper_trading
echo ""

echo "9. CHECK FOR BALANCE/CAPITAL:"
redis-cli KEYS "*balance*"
redis-cli KEYS "*capital*"
redis-cli KEYS "*equity*"
echo ""

echo "10. CHECK LAST EXECUTION METRICS:"
redis-cli GET execution_metrics

ENDSSH

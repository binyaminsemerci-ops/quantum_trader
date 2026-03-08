#!/bin/bash
set -e
echo '=== STEP 3e: EXIT PIPELINE TEST ==='
BEFORE=$(redis-cli XLEN quantum:stream:trade.intent)
INID=$(redis-cli XADD quantum:stream:exit.intent '*' intent_id PATCH6-P6TEST-03 symbol BTCUSDT action FULL_CLOSE urgency HIGH confidence 0.90 R_net 2.0 source exit_management_agent side LONG quantity 0.001 entry_price 50000 mark_price 52000 loop_id p6verif03)
echo "Injected: $INID"
sleep 3
AFTER=$(redis-cli XLEN quantum:stream:trade.intent)
echo "trade.intent BEFORE=$BEFORE AFTER=$AFTER DELTA=$((AFTER-BEFORE))"
echo '--- LATEST trade.intent entry ---'
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1

echo ''
echo '=== STEP 3f: NO DIRECT apply.plan FROM EMA ==='
echo 'Last apply.plan entry (should come from intent_bridge not EMA):'
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 1

echo ''
echo '=== STEP 4: ROLLBACK TEST ==='
echo "Flag BEFORE DEL: $(redis-cli GET quantum:exit_agent:active_flag) (TTL=$(redis-cli TTL quantum:exit_agent:active_flag))"
redis-cli DEL quantum:exit_agent:active_flag
echo 'Deleted. Checking immediately (within EMA tick window)...'
sleep 2
V2=$(redis-cli GET quantum:exit_agent:active_flag)
T2=$(redis-cli TTL quantum:exit_agent:active_flag)
echo "Flag at +2s: '$V2' TTL=$T2"

sleep 6
V8=$(redis-cli GET quantum:exit_agent:active_flag)
T8=$(redis-cli TTL quantum:exit_agent:active_flag)
echo "Flag at +8s (EMA should have renewed): '$V8' TTL=$T8"

echo ''
echo '=== STEP 5: RESTORATION CHECK ==='
echo 'EMA ownership log (last 3 lines):'
journalctl -u quantum-exit-management-agent -n 15 --no-pager | grep -E 'OWNERSHIP_FLAG' | tail -3
echo 'DONE'

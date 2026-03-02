#!/bin/bash
echo "=== Siste gjenværende pos key ==="
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null

echo ""
echo "=== BNB er fortsatt åpen på Binance — force close via intent ==="
# Opprett en manuell close plan for BNB direkte til harvest.intent
plan_id=$(python3 -c "import uuid; print(uuid.uuid4().hex)" 2>/dev/null)
ts=$(date +%s)
redis-cli XADD quantum:stream:harvest.intent "*" \
    plan_id "$plan_id" \
    symbol BNBUSDT \
    side BUY \
    action FULL_CLOSE_PROPOSED \
    decision EXECUTE \
    close_qty 0.17 \
    reduceOnly true \
    source manual_close \
    timestamp "$ts" \
    reason "manual_all_positions_closed" \
    2>/dev/null
echo "  Harvest close plan sendt for BNB (plan_id=$plan_id)"

echo ""
echo "=== Vent 10s og sjekk om BNB er lukket ==="
sleep 10
echo "BNB_snapshot=$(redis-cli HGET quantum:position:snapshot:BNBUSDT position_amt 2>/dev/null)"
echo "BNB_age=$(($(date +%s) - $(redis-cli HGET quantum:position:snapshot:BNBUSDT ts_epoch 2>/dev/null)))s"

echo ""
echo "=== Intent executor: siste executions ==="
journalctl -u quantum-intent-executor.service -n 8 --no-pager 2>/dev/null | grep -E "BNB|execut.*True|🚀|FILLED"

echo ""
echo "=== Delete ADAUSDT cooldown ==="
redis-cli DEL quantum:cooldown:last_exec_ts:ADAUSDT 2>/dev/null
echo "  Deleted"

echo ""
echo "=== FINAL STATE ==="
echo "pos_keys=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | wc -l)"
echo "cooldowns=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)"
echo "BNB=$(redis-cli HGET quantum:position:snapshot:BNBUSDT position_amt 2>/dev/null)"
echo "BTC=$(redis-cli HGET quantum:position:snapshot:BTCUSDT position_amt 2>/dev/null)"
echo "ETH=$(redis-cli HGET quantum:position:snapshot:ETHUSDT position_amt 2>/dev/null)"
echo "SOL=$(redis-cli HGET quantum:position:snapshot:SOLUSDT position_amt 2>/dev/null)"
echo "ADA=$(redis-cli HGET quantum:position:snapshot:ADAUSDT position_amt 2>/dev/null)"

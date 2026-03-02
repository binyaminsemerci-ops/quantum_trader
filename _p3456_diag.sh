#!/bin/bash
echo "=== P3a: GHOST SLOTS I HARVEST-V2 ==="
echo ""
echo "--- HV2 slot nøkler (alle typer) ---"
redis-cli KEYS 'quantum:hv2:slot:*' 2>/dev/null | wc -l
echo "Eksempler:"
redis-cli KEYS 'quantum:hv2:slot:*' 2>/dev/null | head -10

echo ""
echo "--- HV2 position-feed nøkler ---"
redis-cli KEYS 'quantum:hv2:*' 2>/dev/null | head -20

echo ""
echo "--- HV2 kildekode: SKIP_INVALID_RISK ---"
grep -n "SKIP_INVALID_RISK\|skipped_invalid\|invalid_risk\|entry_risk" \
    /home/qt/quantum_trader/microservices/harvest_v2/*.py \
    /home/qt/quantum_trader/microservices/harvest_v2/**/*.py 2>/dev/null | head -20

echo ""
echo "--- HV2: finn kildekode ---"
find /home/qt/quantum_trader -name "*.py" 2>/dev/null | xargs grep -l "SKIP_INVALID_RISK" 2>/dev/null | head -5
find /opt/quantum -name "*.py" 2>/dev/null | xargs grep -l "SKIP_INVALID_RISK" 2>/dev/null | head -5

echo ""
echo "--- HV2: position feed kilde ---"
journalctl -u quantum-harvest-v2.service -n 5 --no-pager 2>/dev/null | head -5

echo ""
echo "=== P3b: STALE HARVEST PROPOSALS ==="
echo ""
echo "--- Alle harvest proposals og positions ---"
for key in $(redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | sort); do
    SYM=$(echo $key | sed 's/quantum:harvest:proposal://')
    ACTION=$(redis-cli HGET "$key" harvest_action 2>/dev/null)
    RNET=$(redis-cli HGET "$key" R_net 2>/dev/null)
    EPOCH=$(redis-cli HGET "$key" last_update_epoch 2>/dev/null)
    AGE_S=$(($(date +%s) - ${EPOCH:-0}))
    POS_EXISTS=$(redis-cli EXISTS "quantum:position:$SYM" 2>/dev/null)
    echo "  $SYM: action=$ACTION R_net=$RNET age=${AGE_S}s pos_exists=$POS_EXISTS"
done

echo ""
echo "=== P3c: OPUSDT/XRPUSDT - har de posisjoner? ==="
redis-cli HGETALL quantum:position:OPUSDT 2>/dev/null
redis-cli HGETALL quantum:position:XRPUSDT 2>/dev/null

echo ""
echo "=== P6: WRONGTYPE — PERMIT NØKLER ==="
echo ""
echo "--- Type på permit-nøkler ---"
redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | head -5 | while read k; do
    TYPE=$(redis-cli TYPE "$k" 2>/dev/null)
    TTL=$(redis-cli TTL "$k" 2>/dev/null)
    VAL=$(redis-cli GET "$k" 2>/dev/null || redis-cli HGETALL "$k" 2>/dev/null | head -4)
    echo "  $k TYPE=$TYPE TTL=$TTL val=$VAL"
done

echo ""
echo "--- Intent-executor: permit kode ---"
grep -n "_wait_for_permit\|hgetall.*permit\|permit.*hgetall\|permit_data" \
    /home/qt/quantum_trader/microservices/intent_executor/main.py 2>/dev/null | head -20

echo ""
echo "--- auto_permit_p33.py type av nøkkel ---"
grep -n "SET\|HSET\|xadd\|set\|hset" \
    /opt/quantum/scripts/auto_permit_p33.py 2>/dev/null | head -20

echo ""
echo "=== P5: BNB PARTIAL_25 qty=0 ==="
echo ""
echo "--- Governor P3.2 qty-beregning ---"
journalctl -u quantum-governor.service -n 20 --no-pager 2>/dev/null | \
    grep -iE "qty|notional|bnb|partial" | tail -10

echo ""
echo "--- BNB posisjon i Redis ---"
redis-cli HGETALL quantum:position:BNBUSDT 2>/dev/null

echo ""
echo "--- Governor kildekode: close_qty ---"
grep -n "close_qty\|PARTIAL_25\|partial_25" \
    /home/qt/quantum_trader/microservices/governor/*.py 2>/dev/null | head -20

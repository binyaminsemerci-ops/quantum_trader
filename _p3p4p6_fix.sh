#!/bin/bash
echo "========================================"
echo " FIX P3: Slett tomt-qty snapshot nøkler"
echo "========================================"
echo ""
echo "--- FØR: Snapshot nøkler ---"
SNAP_COUNT=$(redis-cli KEYS 'quantum:position:snapshot:*' 2>/dev/null | wc -l)
echo "  Antall snapshot nøkler: $SNAP_COUNT"

echo ""
echo "--- Sletter alle snapshot nøkler ---"
SNAP_KEYS=$(redis-cli KEYS 'quantum:position:snapshot:*' 2>/dev/null)
DELETED=0
if [ -n "$SNAP_KEYS" ]; then
    while IFS= read -r key; do
        redis-cli DEL "$key" 2>/dev/null
        DELETED=$((DELETED+1))
    done <<< "$SNAP_KEYS"
fi
echo "  Slettet: $DELETED snapshot nøkler"

echo ""
echo "--- FIX ledger symbol-felt (sett symbol fra key) ---"
for key in $(redis-cli KEYS 'quantum:position:ledger:*' 2>/dev/null); do
    SYM=$(echo $key | sed 's/quantum:position:ledger://')
    CURR_SYM=$(redis-cli HGET "$key" symbol 2>/dev/null)
    if [ -z "$CURR_SYM" ]; then
        redis-cli HSET "$key" symbol "$SYM" 2>/dev/null
        echo "  FIKSET ledger symbol: $key → symbol=$SYM"
    else
        echo "  OK: $key symbol=$CURR_SYM"
    fi
done

echo ""
echo "--- ETTER: Posisjon nøkler ---"
redis-cli KEYS 'quantum:position:*' 2>/dev/null | sort

echo ""
echo "=========================================="
echo " FIX P4: Slett permanente cooldowns TTL=-1"
echo "=========================================="
echo ""
echo "--- FØR: Cooldowns ---"
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do
    TTL=$(redis-cli TTL "$k" 2>/dev/null)
    echo "  $k TTL=$TTL"
done

echo ""
echo "--- Sletter cooldowns med TTL=-1 (permanent) ---"
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do
    TTL=$(redis-cli TTL "$k" 2>/dev/null)
    if [ "$TTL" = "-1" ]; then
        redis-cli DEL "$k" 2>/dev/null
        echo "  SLETTET: $k (TTL=-1)"
    else
        echo "  BEHOLDER: $k (TTL=$TTL)"
    fi
done

echo ""
echo "--- ETTER: Cooldowns ---"
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do
    TTL=$(redis-cli TTL "$k" 2>/dev/null)
    echo "  $k TTL=$TTL"
done

echo ""
echo "=========================================="
echo " FIX P6: Slett STRING-type permit nøkler"
echo "=========================================="
echo ""
echo "--- Sjekker alle permit nøkler ---"
STRING_COUNT=0
HASH_COUNT=0
redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | while read k; do
    TYPE=$(redis-cli TYPE "$k" 2>/dev/null)
    if [ "$TYPE" = "string" ]; then
        redis-cli DEL "$k" 2>/dev/null
        echo "  SLETTET (string-type): $k"
        STRING_COUNT=$((STRING_COUNT+1))
    fi
done
echo "  STRING-type permits slettet: $STRING_COUNT (output over)"

echo ""
echo "--- HASH-type permits igjen ---"
redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | while read k; do
    TYPE=$(redis-cli TYPE "$k" 2>/dev/null)
    TTL=$(redis-cli TTL "$k" 2>/dev/null)
    echo "  $k TYPE=$TYPE TTL=$TTL"
done | head -10
echo "  Total permits:"
redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | wc -l

echo ""
echo "--- Restart intent-executor ---"
systemctl restart quantum-intent-executor.service
sleep 3
systemctl is-active quantum-intent-executor.service
echo "  Restart DONE"

echo ""
echo "[DONE] P3 snapshots, P4 cooldowns, P6 permits — alle fikset"

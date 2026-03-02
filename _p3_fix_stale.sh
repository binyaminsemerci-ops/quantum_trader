#!/bin/bash
echo "=== P3 FIX STEG 1: Slett stale proposals uten posisjoner ==="
echo ""

echo "--- FØR: Harvest proposals ---"
for key in $(redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | sort); do
    SYM=$(echo $key | sed 's/quantum:harvest:proposal://')
    POS=$(redis-cli EXISTS "quantum:position:$SYM" 2>/dev/null)
    echo "  $SYM: pos_exists=$POS"
done

echo ""
echo "--- Sletter proposals uten posisjoner ---"
for key in $(redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null); do
    SYM=$(echo $key | sed 's/quantum:harvest:proposal://')
    POS=$(redis-cli EXISTS "quantum:position:$SYM" 2>/dev/null)
    if [ "$POS" = "0" ]; then
        redis-cli DEL "$key" 2>/dev/null
        echo "  SLETTET: $key (ingen posisjon)"
    else
        echo "  BEHOLDER: $key (posisjon eksisterer)"
    fi
done

echo ""
echo "--- ETTER: Gjenværende proposals ---"
redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | sort | while read k; do
    SYM=$(echo $k | sed 's/quantum:harvest:proposal://')
    ACTION=$(redis-cli HGET "$k" harvest_action 2>/dev/null)
    RNET=$(redis-cli HGET "$k" R_net 2>/dev/null)
    echo "  $SYM action=$ACTION R_net=$RNET"
done

echo ""
echo "--- Slett tilhørende apply:dedupe nøkler for slettede symboler ---"
for SYM in OPUSDT XRPUSDT; do
    KEYS=$(redis-cli KEYS "quantum:apply:dedupe:*${SYM}*" 2>/dev/null)
    if [ -n "$KEYS" ]; then
        echo "$KEYS" | while read k; do
            redis-cli DEL "$k" 2>/dev/null
            echo "  SLETTET dedupe: $k"
        done
    fi
done

echo ""
echo "[DONE] P3 stale proposals fjernet"

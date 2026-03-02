#!/bin/bash
echo "=== P3 DIAGNOSE: Ghost slot entries ==="
echo ""

echo "--- Alle position-ledger nøkler ---"
redis-cli KEYS 'quantum:position:ledger:*' 2>/dev/null | sort

echo ""
echo "--- Alle posisjon nøkler (inkl alt) ---"
redis-cli KEYS 'quantum:position:*' 2>/dev/null | sort

echo ""
echo "--- HV2 kode: hvorfra leser den posisjoner? ---"
grep -n "position\|ledger\|slot\|KEYS\|SCAN\|hgetall\|HGETALL" \
    /home/qt/quantum_trader/microservices/harvest_v2/harvest_v2.py 2>/dev/null | \
    grep -v "^#" | head -30

echo ""
echo "--- HV2 kode: PositionFeed klasse ---"
grep -n "class Position\|def.*position\|def.*fetch\|def.*load" \
    /home/qt/quantum_trader/microservices/harvest_v2/harvest_v2.py 2>/dev/null | head -20

echo ""
echo "--- HV2 feeds kode (position feed) ---"
find /home/qt/quantum_trader/microservices/harvest_v2 -name "*.py" 2>/dev/null | \
    xargs grep -l "position\|HGETALL\|ledger" 2>/dev/null

echo ""
echo "--- Content av feeds/position.py ---"
cat /home/qt/quantum_trader/microservices/harvest_v2/feeds/position.py 2>/dev/null | head -80

echo ""
echo "--- Sjekk: hva er ledger nøkler? ---"
for key in $(redis-cli KEYS 'quantum:position:ledger:*' 2>/dev/null | head -10); do
    TYPE=$(redis-cli TYPE "$key" 2>/dev/null)
    echo "  $key TYPE=$TYPE"
    if [ "$TYPE" = "hash" ]; then
        SYM=$(redis-cli HGET "$key" symbol 2>/dev/null)
        echo "    symbol='$SYM'"
    fi
done

echo ""
echo "--- Sjekk: tomme symbol i ledger ---"
redis-cli KEYS 'quantum:position:ledger:*' 2>/dev/null | while read key; do
    SYM=$(redis-cli HGET "$key" symbol 2>/dev/null)
    if [ -z "$SYM" ]; then
        echo "  GHOST (tom symbol): $key"
    fi
done | head -20
echo "Total ghost ledger entries:"
redis-cli KEYS 'quantum:position:ledger:*' 2>/dev/null | while read key; do
    SYM=$(redis-cli HGET "$key" symbol 2>/dev/null)
    [ -z "$SYM" ] && echo "ghost"
done | wc -l

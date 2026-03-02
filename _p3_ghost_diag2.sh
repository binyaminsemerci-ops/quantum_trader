#!/bin/bash
echo "=== PositionProvider kildekode ==="
cat /home/qt/quantum_trader/microservices/harvest_v2/feeds/position_provider.py 2>/dev/null

echo ""
echo "=== Snapshot nøkler - innhold sample ==="
for key in $(redis-cli KEYS 'quantum:position:snapshot:*' 2>/dev/null | head -5); do
    SYM=$(echo $key | sed 's/quantum:position:snapshot://')
    QTY=$(redis-cli HGET "$key" position_amt 2>/dev/null || redis-cli HGET "$key" quantity 2>/dev/null)
    RISK=$(redis-cli HGET "$key" entry_risk_usdt 2>/dev/null)
    echo "  $SYM: qty=$QTY entry_risk=$RISK"
done

echo ""
echo "=== Ledger nøkler innhold ==="
for key in $(redis-cli KEYS 'quantum:position:ledger:*' 2>/dev/null); do
    echo "--- $key ---"
    redis-cli HGETALL "$key" 2>/dev/null | paste - - | head -10
done

echo ""
echo "=== ADAUSDT/SOLUSDT posisjoner (mangler fra ledger) ==="
redis-cli EXISTS quantum:position:ledger:ADAUSDT 2>/dev/null
redis-cli EXISTS quantum:position:ledger:SOLUSDT 2>/dev/null
redis-cli HGETALL quantum:position:ADAUSDT 2>/dev/null | paste - -
redis-cli HGETALL quantum:position:SOLUSDT 2>/dev/null | paste - -

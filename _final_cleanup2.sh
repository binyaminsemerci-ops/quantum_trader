#!/bin/bash
echo "=== Remaining pos key ==="
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | while read k; do
    redis-cli HGETALL "$k" 2>/dev/null | head -6
    echo "  -> DEL $k"
    redis-cli DEL "$k" 2>/dev/null
done

echo ""
echo "=== Remaining cooldown ==="
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do
    echo "  $k"
    redis-cli DEL "$k" 2>/dev/null
done

echo ""
echo "=== BNB Binance snapshot (aktuell) ==="
redis-cli HGET quantum:position:snapshot:BNBUSDT position_amt 2>/dev/null
redis-cli HGET quantum:position:snapshot:BNBUSDT ts_epoch 2>/dev/null
now=$(date +%s)
ts=$(redis-cli HGET quantum:position:snapshot:BNBUSDT ts_epoch 2>/dev/null)
echo "BNB snapshot age: $((now - ts))s"

echo ""
echo "=== OBS: Sjekk Binance manuelt om BNB SHORT fortsatt er åpen ==="
echo "BNB position_amt=$(redis-cli HGET quantum:position:snapshot:BNBUSDT position_amt 2>/dev/null)"
echo "Hvis dette IKKE er 0.0 etter ~30s, er BNB SHORT fortsatt åpen på Binance testnet"

echo ""
echo "=== FINAL ==="
echo "pos_keys=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | wc -l)"
echo "cooldowns=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)"

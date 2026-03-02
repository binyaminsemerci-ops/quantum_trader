#!/bin/bash
echo "=== Remaining pos keys ==="
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null

echo ""
echo "=== Remaining cooldowns ==="
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null

echo ""
echo "=== Remaining kelly keys ==="
redis-cli KEYS 'quantum:layer4:sizing:*' 2>/dev/null

echo ""
echo "=== BNB snapshot full ==="
redis-cli HGETALL quantum:position:snapshot:BNBUSDT 2>/dev/null

echo ""
echo "=== BNB Binance snapshot age ==="
ts=$(redis-cli HGET quantum:position:snapshot:BNBUSDT ts_epoch 2>/dev/null)
now=$(date +%s)
age=$((now - ts))
echo "Snapshot age: ${age}s"
echo "(BNB data kan være stale hvis Binance allerede er flat og snapshot ikke er oppdatert ennå)"

echo ""
echo "=== DELETE resterende position keys ==="
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | while read k; do
    echo "  DEL $k"
    redis-cli DEL "$k" 2>/dev/null
done

echo ""
echo "=== DELETE resterende cooldowns ==="
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do
    redis-cli DEL "$k" 2>/dev/null
done
echo "  Cooldowns cleared"

echo ""
echo "=== RESET Kelly til 12 symboler ==="
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    redis-cli HSET "quantum:layer4:sizing:$sym" \
        symbol "$sym" size_usdt 200.0 \
        recommendation MINIMUM_VIABLE_FLOOR \
        kelly_adj 0.04 updated_at "$(date +%s)" 2>/dev/null > /dev/null
done
echo "  Kelly floor satt for 12 symboler"

echo ""
echo "=== FINAL STATE ==="
echo "pos_keys=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | wc -l)"
echo "cooldowns=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)"
echo "kelly=$(redis-cli KEYS 'quantum:layer4:sizing:*' 2>/dev/null | wc -l)"
echo "BNB_snapshot=$(redis-cli HGET quantum:position:snapshot:BNBUSDT position_amt 2>/dev/null)"
echo ""
echo "Sistema er flat og klar for ny trading"

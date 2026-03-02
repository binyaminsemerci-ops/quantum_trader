#!/bin/bash
echo "=== ALLE POSISJONER LUKKET MANUELT — Redis cleanup ==="
echo "Tidspunkt: $(date -u)"

echo ""
echo "=== BEFORE: position keys ==="
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | while read k; do
    side=$(redis-cli HGET "$k" side 2>/dev/null)
    qty=$(redis-cli HGET "$k" quantity 2>/dev/null)
    echo "  $k: $side qty=$qty"
done

echo ""
echo "=== DELETE: alle quantum:position:<SYMBOL> objekter ==="
for sym in BTCUSDT ETHUSDT ADAUSDT SOLUSDT BNBUSDT DOGEUSDT XRPUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    exists=$(redis-cli EXISTS "quantum:position:$sym" 2>/dev/null)
    if [ "$exists" = "1" ]; then
        redis-cli DEL "quantum:position:$sym" 2>/dev/null
        echo "  DELETED quantum:position:$sym"
    fi
done

echo ""
echo "=== DELETE: harvest proposals ==="
for sym in BTCUSDT ETHUSDT ADAUSDT SOLUSDT BNBUSDT DOGEUSDT XRPUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    exists=$(redis-cli EXISTS "quantum:harvest:proposal:$sym" 2>/dev/null)
    if [ "$exists" = "1" ]; then
        redis-cli DEL "quantum:harvest:proposal:$sym" 2>/dev/null
        echo "  DELETED quantum:harvest:proposal:$sym"
    fi
done

echo ""
echo "=== DELETE: ledger positions (stale) ==="
for sym in BTCUSDT ETHUSDT ADAUSDT SOLUSDT BNBUSDT DOGEUSDT XRPUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    exists=$(redis-cli EXISTS "quantum:position:ledger:$sym" 2>/dev/null)
    if [ "$exists" = "1" ]; then
        redis-cli DEL "quantum:position:ledger:$sym" 2>/dev/null
        echo "  DELETED quantum:position:ledger:$sym"
    fi
done

echo ""
echo "=== DELETE: Kelly sizing (reset floor) ==="
redis-cli KEYS 'quantum:layer4:sizing:*' 2>/dev/null | while read k; do
    redis-cli DEL "$k" 2>/dev/null
    echo "  DELETED $k"
done

echo ""
echo "=== DELETE: Active cooldowns (new start, fresh slate) ==="
cooldown_count=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)
echo "  Deleting $cooldown_count cooldown keys..."
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do
    redis-cli DEL "$k" 2>/dev/null
done
echo "  Done"

echo ""
echo "=== DELETE: Executed plan markers (fresh start) ==="
done_count=$(redis-cli KEYS 'quantum:intent_executor:done:*' 2>/dev/null | wc -l)
echo "  Found $done_count done-keys — deleting..."
redis-cli KEYS 'quantum:intent_executor:done:*' 2>/dev/null | while read k; do
    redis-cli DEL "$k" 2>/dev/null
done
echo "  Done"

echo ""
echo "=== DELETE: Apply plan dedup markers ==="
dedup_count=$(redis-cli KEYS 'quantum:dedup:plan:*' 2>/dev/null | wc -l)
echo "  Found $dedup_count dedup-keys — deleting..."
redis-cli KEYS 'quantum:dedup:plan:*' 2>/dev/null | while read k; do
    redis-cli DEL "$k" 2>/dev/null
done

echo ""
echo "=== DELETE: Risk/daily PnL counters (fresh day start) ==="
redis-cli DEL quantum:risk:daily_pnl 2>/dev/null && echo "  DELETED quantum:risk:daily_pnl"
redis-cli DEL quantum:risk:consecutive_losses 2>/dev/null && echo "  DELETED quantum:risk:consecutive_losses"
redis-cli DEL quantum:risk:daily_limit_hit 2>/dev/null && echo "  DELETED quantum:risk:daily_limit_hit"

echo ""
echo "=== REFRESH: Restart key services to pick up flat state ==="
systemctl restart quantum-apply-layer.service 2>/dev/null
sleep 1
systemctl restart quantum-intent-executor.service 2>/dev/null
sleep 1
echo "  apply-layer: $(systemctl is-active quantum-apply-layer.service 2>/dev/null)"
echo "  intent-executor: $(systemctl is-active quantum-intent-executor.service 2>/dev/null)"

echo ""
echo "=== REFRESH: Kelly sizing floor for all 12 symbols ==="
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    redis-cli HSET "quantum:layer4:sizing:$sym" \
        symbol "$sym" \
        size_usdt 200.0 \
        recommendation MINIMUM_VIABLE_FLOOR \
        kelly_adj 0.04 \
        updated_at "$(date +%s)" \
        2>/dev/null > /dev/null
done
echo "  Kelly floor $200 satt for alle 12 symboler"

echo ""
echo "=== VERIFY: State after cleanup ==="
pos_count=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | wc -l)
cooldown_left=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)
proposal_left=$(redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | wc -l)
kelly_count=$(redis-cli KEYS 'quantum:layer4:sizing:*' 2>/dev/null | wc -l)
echo "  Open positions in Redis: $pos_count (forventet: 0)"
echo "  Active cooldowns: $cooldown_left (forventet: 0)"
echo "  Harvest proposals: $proposal_left (forventet: 0)"
echo "  Kelly sizing keys: $kelly_count (forventet: 12)"

echo ""
echo "=== Binance snapshot bekrefter flat? ==="
for sym in BTCUSDT ETHUSDT ADAUSDT SOLUSDT BNBUSDT; do
    amt=$(redis-cli HGET "quantum:position:snapshot:$sym" position_amt 2>/dev/null)
    echo "  $sym binance_amt=$amt (forventet: 0.0)"
done

echo ""
echo "============================================================"
echo " CLEANUP KOMPLETT — Systemet er klart for ny start"
echo "============================================================"

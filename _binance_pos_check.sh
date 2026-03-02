#!/bin/bash
echo "=== Latest apply.plan (FULL - first few fields) ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 2 2>/dev/null | head -40

echo ""
echo "=== Binance positions right now (from snapshot) ==="
for sym in BTCUSDT ETHUSDT SOLUSDT ADAUSDT BNBUSDT; do
    amt=$(redis-cli HGET quantum:position:snapshot:$sym position_amt 2>/dev/null)
    side=$(redis-cli HGET quantum:position:snapshot:$sym side 2>/dev/null)
    upnl=$(redis-cli HGET quantum:position:snapshot:$sym unrealized_pnl 2>/dev/null)
    ts=$(redis-cli HGET quantum:position:snapshot:$sym ts_epoch 2>/dev/null)
    echo "  $sym: $side amt=$amt upnl=$upnl (snapshot_ts=$ts)"
done

echo ""
echo "=== Harvest_v2 close plans (source=harvest_v2) last 5 ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 50 2>/dev/null | python3 -c "
import sys
lines = sys.stdin.read().split('\n')
entries = []
current = {}
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.endswith('-0') or line.endswith('-1'):  # message id
        if current:
            entries.append(current)
        current = {'id': line}
    elif line and i+1 < len(lines):
        key = line.strip()
        val = lines[i+1].strip() if i+1 < len(lines) else ''
        current[key] = val
        i += 1
    i += 1
if current:
    entries.append(current)
count = 0
for e in entries:
    if e.get('source') == 'harvest_v2' or e.get('action','').startswith('FULL_CLOSE'):
        if count < 5:
            print(f'  symbol={e.get(\"symbol\")} action={e.get(\"action\")} close_qty={e.get(\"close_qty\")} source={e.get(\"source\")} decision={e.get(\"decision\")}')
            count += 1
" 2>/dev/null

echo ""
echo "=== MOST RECENT executions ==="
journalctl -u quantum-intent-executor.service --since '5 minutes ago' --no-pager 2>/dev/null | grep "execut.*True" | tail -10

echo ""
echo "=== TOTAL PnL summary ==="
total=0
for sym in BTCUSDT ETHUSDT SOLUSDT ADAUSDT; do
    upnl=$(redis-cli HGET quantum:position:snapshot:$sym unrealized_pnl 2>/dev/null)
    echo "  $sym upnl=$upnl"
done

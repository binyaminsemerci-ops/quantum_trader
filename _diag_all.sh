#!/bin/bash
echo "=== ADA STATUS ==="
redis-cli HGETALL quantum:position:ADAUSDT 2>/dev/null
echo "--- snapshot ---"
redis-cli HGETALL quantum:position:snapshot:ADAUSDT 2>/dev/null | grep -E "position_amt|side|unrealized|ts_epoch"
echo "--- harvest proposal ---"
redis-cli HGETALL quantum:harvest:proposal:ADAUSDT 2>/dev/null
echo "--- latest ADA plan in apply.plan ---"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 2>/dev/null | python3 -c "
import sys
data=sys.stdin.read().split('\n')
i=0
found=False
while i<len(data):
    if 'ADAUSDT' in data[i] and not found:
        start=max(0,i-10)
        for j in range(start,min(len(data),i+25)):
            print(data[j])
        found=True
    i+=1
" 2>/dev/null

echo ""
echo "=== EXCHANGE.RAW — BTCUSDT check ==="
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 500 2>/dev/null | python3 -c "
import sys
data = sys.stdin.read().split('\n')
btc_count = 0
i = 0
while i < len(data):
    if data[i].strip() == 'BTCUSDT':
        btc_count += 1
    i += 1
print(f'BTCUSDT entries in last 500: {btc_count}')
" 2>/dev/null
redis-cli XLEN quantum:stream:exchange.raw 2>/dev/null

echo ""
echo "=== Exchange stream bridge logs ==="
journalctl -u quantum-exchange-stream-bridge.service -n 20 --no-pager 2>/dev/null | tail -20

echo ""
echo "=== Exchange stream bridge service file ==="
cat /etc/systemd/system/quantum-exchange-stream-bridge.service 2>/dev/null | head -20

echo ""
echo "=== P3.5 guard source ==="
grep -n "kelly_200usdt\|kelly.*200\|200.*usdt\|CLOSE\|close\|action_hold\|action_normalized\|partial_25" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -40

echo ""
echo "=== Intent executor: section around P3.5 guard decision ==="
grep -n "P3.5\|p35\|p3_5\|guard_decision\|GUARD.*SKIP\|SKIP.*GUARD\|kelly_200\|_guard\b" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -30

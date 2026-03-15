#!/bin/bash
echo "=== 1INCHUSDT position ==="
redis-cli hgetall quantum:state:positions:1INCHUSDT 2>/dev/null

echo ""
echo "=== apply.result - søk etter AAVEUSDT, ACEUSDT, 1INCH ==="
redis-cli xrevrange quantum:stream:apply.result + - COUNT 500 2>/dev/null | python3 -c "
import sys, re
entries = []
cid, cf, fn = None, {}, None
for l in sys.stdin:
    l = l.strip()
    if not l: continue
    if re.match(r'^\d+-\d+$', l):
        if cid: entries.append(cf)
        cid, cf, fn = l, {}, None
    elif fn is None: fn = l
    else: cf[fn] = l; fn = None
if cid: entries.append(cf)
hits = [e for e in entries if e.get('symbol','') in ('AAVEUSDT','ACEUSDT','1INCHUSDT')]
print(f'Hits for AAVEUSDT/ACEUSDT/1INCHUSDT: {len(hits)}')
for e in hits:
    print(f'  {e.get(\"symbol\")} action={e.get(\"action\")} executed={e.get(\"executed\")} error={e.get(\"error\")}')
# Also show any executed=True
true_count = sum(1 for e in entries if e.get('executed') == 'True')
print(f'Total executed=True in last 500: {true_count}')
"

echo ""
echo "=== apply.result - executed=True siste 50 ==="
redis-cli xrevrange quantum:stream:apply.result + - COUNT 50 2>/dev/null | grep -B1 "True$" | head -20

echo ""
echo "=== apply_layer recent logs for ENTRY ==="
journalctl -u quantum-apply-layer --no-pager --since "10 minutes ago" -q 2>/dev/null | grep -E "ENTRY|1INCH|AAVE|ACE" | tail -20

echo ""
echo "=== execution.fills stream ==="
redis-cli xrevrange quantum:stream:execution.fills + - COUNT 5 2>/dev/null | head -30
